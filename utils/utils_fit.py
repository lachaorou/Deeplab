import os
import numpy as np

import torch
from models.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss,
                                      weights_init, get_lr_scheduler, set_optimizer_lr)
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
    fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch

        # Debug: 打印前2个batch的标签和模型输出unique值
        # if iteration < 2 and local_rank == 0:
        #     print(f"[Debug][Train] imgs shape: {imgs.shape}, pngs shape: {pngs.shape}, labels shape: {labels.shape}")
        #     print(f"[Debug][Train] 标签 unique: {torch.unique(pngs)}")
        #     outputs_debug = model_train(imgs)
        #     if hasattr(outputs_debug, 'detach'):
        #         out_np = outputs_debug.detach().cpu().numpy()
        #         print(f"[Debug][Train] 预测 unique: {np.unique(out_np)}")

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            # Debug: 打印预测shape和unique值
            if iteration < 2 and local_rank == 0:
                pass
            #----------------------#
            #   计算损失
            #----------------------#
            # 修正：CE_Loss/Focal_Loss 只用 pngs，Dice_loss 用 labels
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
            # Debug: 打印优化器参数统计
            if iteration < 2 and local_rank == 0:
                pass
        else:
            from torch.cuda.amp import autocast
            with torch.amp.autocast('cuda'):
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                if iteration < 2 and local_rank == 0:
                    pass
                #----------------------#
                #   计算损失
                #----------------------#
                # 修正：CE_Loss/Focal_Loss 只用 pngs，Dice_loss 用 labels
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)
                    
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if iteration < 2 and local_rank == 0:
                for i, group in enumerate(optimizer.param_groups):
                    for j, p in enumerate(group['params']):
                        if p.requires_grad and p.data is not None:
                            pass  # 保证语法正确
                        break
                    break

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        # Debug: 打印验证集输入/标签/onehot标签的shape和unique值
        # if iteration < 2 and local_rank == 0:
        #     print(f"[Debug][Val] imgs shape: {imgs.shape}, pngs shape: {pngs.shape}, labels shape: {labels.shape}")
        #     print(f"[Debug][Val] 标签 unique: {torch.unique(pngs)}")
        #     with torch.no_grad():
        #         outputs = model_train(imgs)
        #         if hasattr(outputs, 'detach'):
        #             out_np = outputs.detach().cpu().numpy()
        #             print(f"[Debug][Val] 预测 unique: {np.unique(out_np)}")
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
            outputs     = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            val_loss += loss.item()
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print(f"[Debug] fit_one_epoch: 准备调用 eval_callback.on_epoch_end, epoch={epoch + 1}")
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            # 自动记录参数到config.txt和experiment_records.csv
            try:
                from scripts.train import save_experiment_record
                import datetime
                # 构造参数字典
                param_dict = {
                    'backbone': getattr(model, 'backbone', 'unknown') if hasattr(model, 'backbone') else 'unknown',
                    'input_shape': list(model.input_shape) if hasattr(model, 'input_shape') else '',
                    'num_classes': num_classes,
                    'save_dir': save_dir,
                    'VOCdevkit_path': '',  # 可根据实际传入
                    'token_length': getattr(model, 'token_length', ''),
                    'train_time': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                }
                best_weight_path = os.path.join(save_dir, "best_epoch_weights.pth")
                save_experiment_record(param_dict, best_weight_path)
            except Exception as e:
                print(f"[Warning] 自动记录实验参数失败: {e}")
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))