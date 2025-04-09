import numpy as np
import torch
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import h5py
from ode import *
from lsf import *
import os
import matplotlib.pyplot as plt
import re



def model_train(n_epoch, n_batch, ndim, device, importance_sampler, prior, func, func2, logdet_weight_target, optimizer, verbose, print_file, save, path_save, save_name):
    loss_list = []
    loss_prior_list = []
    loss_data_list = []
    logdet_list = []
    loss2_list = []
    pf_list = []
    pf1_list = []
    scaling_val = []
    
    beta2_start = 20
    beta1_start = 1

    alpha_start = 1
    alpha_end = 10
	
    T_cool = 0.1*n_epoch
    scheduler = MultiStepLR(optimizer, milestones=[T_cool], gamma=0.9) #intially 1.0  d d


    #pretraining steps:
    for _ in range(10):
        z_sample = torch.randn(n_batch, ndim).to(device=device)
        theta_sample, logdet = importance_sampler(z_sample)
        loss = -torch.mean(logdet)

        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(importance_sampler.parameters(), 0.01)
        optimizer.step()

    
    file_name = "training_log.txt"
    with open(file_name, "w") as file:
        file.write("Epoch\tLoss\n") 

    ########loss = 1e10     ######

    for k in range(n_epoch):

        if k+1 <= n_epoch - T_cool:
            alpha = alpha_start + (alpha_end - alpha_start)*((k+1) / (n_epoch-T_cool))
            beta1 = beta1_start + (1.0 - beta1_start)*((k+1) / (n_epoch-T_cool))
            beta2 = beta2_start - (beta2_start - logdet_weight_target)*((k+1) / (n_epoch-T_cool))
        else:
            alpha = alpha_end
            beta1 = 1.0   ###############  what are these??
            beta2 = logdet_weight_target  ############# what are these ??
        
        z_sample = torch.randn(n_batch, ndim).to(device=device)    ########### initially sampled z 

        
        ########################

        theta_sample, logdet1 = importance_sampler(z_sample)

        logdet = torch.squeeze(logdet1)
        
        loss_data = torch.log(func(theta_sample, alpha))
        loss_prior = prior.log_prob(theta_sample)
            
        loss = -1*torch.mean(loss_data) - beta1*torch.mean(loss_prior) - beta2*torch.mean(logdet)
        loss2 = torch.var(loss_data + loss_prior + logdet - prior.log_prob(z_sample))

        #print(loss2)


        pf = torch.mean((func2(theta_sample).squeeze().to(device))*torch.exp((loss_prior + logdet - prior.log_prob(z_sample))))
        pf1 = torch.mean(torch.exp(loss_data + loss_prior + logdet - prior.log_prob(z_sample)))

        optimizer.zero_grad()
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(importance_sampler.parameters(), 0.01)
        optimizer.step()
        scheduler.step()

        loss_list.append(loss.detach().cpu().numpy())
        loss_data_list.append(torch.mean(loss_data).detach().cpu().numpy())
        loss_prior_list.append(torch.mean(loss_prior).detach().cpu().numpy())
        logdet_list.append(-torch.mean(logdet).detach().cpu().numpy())
        loss2_list.append(loss2.detach().cpu().numpy())
        pf_list.append(pf.detach().cpu().numpy())
        pf1_list.append(pf1.detach().cpu().numpy())

        #print("epoch:", k, " loss:", loss.detach().cpu().numpy())
        #if verbose:
        print(f"epoch: {k:}, loss: {loss_list[-1]:.2f}, loss2: {loss2_list[-1]:.2f}, pf:{pf_list[-1]:1.2e}, pf1:{pf1_list[-1]:1.2e}, loss data: {loss_data_list[-1]:.2f}, loss prior: {loss_prior_list[-1]:.2f}, logdet: {logdet_list[-1]:.2f}")

        with open(file_name, "a") as file:
            file.write(f"{k:}\t{loss_list[-1]:.2f}\n")



        # output_image = "epoch_vs_loss.png"  # Output image file name

        # epochs = []
        # loss = []

        # # Read data from file
        # with open(file_name, 'r') as file:
        #     lines = file.readlines()
        #     for line in lines[1:]:
        #         epoch, loss_value = map(float, line.split())
        #         epochs.append(int(epoch))
        #         loss.append(loss_value)

        # # Plotting
        # plt.figure(figsize=(10, 6))
        # plt.plot(epochs, loss, marker='o', linestyle='-', color='b', label='Loss')
        # plt.title('Epoch vs Loss', fontsize=14)
        # plt.xlabel('Epoch', fontsize=12)
        # plt.ylabel('Loss', fontsize=12)
        # plt.axhline(y=0, color='r', linestyle='--', label='Zero Loss Line')
        # plt.grid(True, linestyle='--', alpha=0.7)
        # plt.legend(fontsize=12)
        # plt.tight_layout()

        # plt.savefig(output_image)

        # # Show the plot




    # save model
    if save:
        torch.save(importance_sampler.state_dict(),path_save + save_name +'.pt')

        for index_seed in range (100):   ################## inference with varying seed values
                
            print(save_name, "save name!!")
            save_name = re.sub(r"Seed_\d+_", f"Seed_{index_seed}_", save_name)
            print(save_name, "updated name !!!!")

            # "Seed_2_flow_60_batch_2_epochs_1_logdet_2.0_ndim_2_lr_0.001" <=save name

            print("calculating prob...")

            n_samples = 1000  # n_samples = 5000   1000    #1000
            z_sample = torch.randn(n_samples, ndim).to(device=device)
            theta_sample, logdet1 = importance_sampler(z_sample)

            logdet = torch.squeeze(logdet1)
            loss_data = torch.log(func(theta_sample, alpha_end))
            loss_prior = prior.log_prob(theta_sample)


            print("theta_sample device:", theta_sample.device)
            print("logdet device:", logdet.device)
            print("loss_prior device:", loss_prior.device)


            func2_output = func2(theta_sample).to(device)
            prior_log_prob = prior.log_prob(z_sample).to(device)

            pf_ = torch.mean(
                func2_output.squeeze() * torch.exp((loss_prior + logdet - prior_log_prob))
            )

            # pf_ = torch.mean((func2(theta_sample).squeeze())*torch.exp(loss_prior + logdet - prior.log_prob(z_sample)))
            pf1_ = torch.mean(torch.exp(loss_data + loss_prior + logdet - prior.log_prob(z_sample)))

            print(f"pf =  {pf_:1.2e}")
            print(f"pf1 =  {pf1_:1.2e}")

            with h5py.File(path_save + save_name +'.h5', 'w') as f:
                f.create_dataset('total', data=np.array(loss_list))
                f.create_dataset('data', data=np.array(loss_data_list))
                f.create_dataset('logdet', data=np.array(logdet_list))
                f.create_dataset('prior', data=np.array(loss_prior_list))
                f.create_dataset('var', data=np.array(loss2_list))
                f.create_dataset('pf', data=np.array(pf_list))
                f.create_dataset('pf1', data=np.array(pf1_list))
                f.create_dataset('est_pf', data=pf_.detach().cpu().numpy())
                f.create_dataset('est_pf1', data=pf1_.detach().cpu().numpy())
                f.create_dataset('samples', data=theta_sample.detach().cpu().numpy())

        
if __name__=='__main__':
    print('This function stores all the training scripts.')
