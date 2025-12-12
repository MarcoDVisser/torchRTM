"""
torchrtm.utils.torch_utils
--------------------------

General-purpose torch utilities.
"""

import torch
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm
#from torchrtm.models import prosail
from torchrtm.utils import normalize_parameters
from torchrtm.leaf.prospect import prospect5b, prospectd, prospectpro


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Torchlut(model = 'prospect5b', table_size=500000, std=0, batch=10000, wavelength=None,sensor_name = 'LANDSAT4-TM',sail_prospect = 'prospectd',use_atom = False,para_addr=None):
    # Initialize Latin Hypercube Sampler
    
    batch = min(table_size,batch)
    import math
    d = 18

    #from torchrtm.atmosphere.smac import toc_to_toa


    if use_atom == True:
        param_type = 'atom'
        from torchrtm.models import prosail
        simulator = prosail
        from torchrtm.atmosphere.smac import smac
        from torchrtm.data_loader import load_smac_sensor
        from torchrtm.atmosphere.smac import toc_to_toa

    else:
        if model == 'prosail':
            from torchrtm.models import prosail
            simulator = prosail
        elif model == 'prospect5b' or model == 'prospect5':
            simulator = prospect5b
        elif model == 'prospectd' ormodel == 'prospect5d'  :
            simulator = prospectd
        elif model == 'prospectpro':
            simulator = prospectpro
        param_type= simulator.__name__
    print(param_type)
    sampler = qmc.LatinHypercube(d=d)
    samples = sampler.random(n=table_size)
    ref_list = []
    para_list = []
    # Determine the total number of iterations
    num_iterations = max(1, math.ceil(table_size / batch))
    # Loop to process samples in batches with a progress bar
    print_flag = True
    for ii_index in tqdm(range(num_iterations), desc="Processing Batches"):
        test_num = batch
        # Generate a subset of samples and convert to PyTorch tensor
        many_paras = torch.tensor(samples[batch * ii_index:batch * (ii_index + 1), :]).to(torch.float32).to(device)
        #print(many_paras[0])
        # Example normalization function placeholder; replace with actual normalization if needed
        many_paras[:,:17] = normalize_parameters(many_paras[:,:17],param_type='prosail', fitting=False,para_addr=para_addr)
        #print(many_paras[0])

        #real_paras = normlization_torch(many_paras, param_type=param_type, fitting=False)
        # Pass through prospect5b model
        if param_type in ['prospectd', 'prospect5b','prospectpro']:
            # For PROSPECT models: typically (leaf_params, LAI) or similar structure
            if param_type == 'prospectd':
                spec_data = simulator(many_paras[:, 9:15], many_paras[:, 8])[0]
                para_list.append(many_paras[:,8:15].cpu().numpy())

            elif param_type == 'prospect5b':
                spec_data = simulator(many_paras[:, 9:14], many_paras[:, 8][0])
                para_list.append(many_paras[:,8:14].cpu().numpy())
            elif param_type == 'prospectpro':
                spec_data = simulator(many_paras[:, [9,10,11,12,14,15,16]], many_paras[:, 8])[0]
                para_list.append(many_paras[:,8:17].cpu().numpy())
        else:
            if param_type == 'atom':
                  many_paras[:,-3:] = normalize_parameters(many_paras[:,-3:],param_type=param_type, fitting=False,para_addr=para_addr)

            lai = many_paras[:,0]
            LIDFa = many_paras[:,1]
            LIDFb = many_paras[:,2]
            q = many_paras[:,3]
            tts = many_paras[:,4]
            tto = many_paras[:,5]
            psi = many_paras[:,6]
            psoil = many_paras[:,7]
            N = many_paras[:,8]
            Cab = many_paras[:,9]
            Car = many_paras[:,10]
            Cbrown = many_paras[:,11]
            Cw = many_paras[:,12]
            Cm = many_paras[:,13]
            Canth = many_paras[:,14]
            prot = many_paras[:,15]
            Cbc = many_paras[:,16]



            alpha = torch.tensor(40).to(device).expand([test_num])
            tran_alpha = alpha.clone()
            if sail_prospect == 'prospectd':
                traits = torch.stack([Cab,Car,Cbrown,Cw,Cm,Canth],dim = 1)
                spec_data = simulator(traits,N,LIDFa,LIDFb,lai,q,tts,tto,psi,tran_alpha,psoil,batch_size=1,prospect_type='prospect5d',lidtype=2)
            elif sail_prospect == 'prospect5':
                traits = torch.stack([Cab,Car,Cbrown,Cw,Cm],dim = 1)
                spec_data = simulator(traits,N,LIDFa,LIDFb,lai,q,tts,tto,psi,tran_alpha,psoil,batch_size=1,prospect_type='prospect5b',lidtype=2)
            elif sail_prospect == 'prospectpro':
                traits = torch.stack([Cab,Car,Cbrown,Cw,Canth,prot,Cbc],dim = 1)
                spec_data = simulator(traits,N,LIDFa,LIDFb,lai,q,tts,tto,psi,tran_alpha,psoil,batch_size=1,prospect_type='prospectpro',lidtype=2)
            if param_type == 'atom':
                aot550 = many_paras[:,-3]
                uo3 = many_paras[:,-2]
                uh2o = many_paras[:,-1]
                coefs,sm_wl = load_smac_sensor(sensor_name.split('.')[0])
                Ta_s, Ta_o, T_g, ra_dd, ra_so, ta_ss, ta_sd, ta_oo, ta_do = smac(tts,tto,psi,coefs)
                # return to the R_TOA
                #sm_wl = np.arange(400,2501)
                #print(spec_data.permute(0, 2, 1).shape,ta_ss.shape, ta_sd.shape, ta_oo.shape, ta_do.shape) for debug
                R_TOC, R_TOA = toc_to_toa(spec_data.permute(0, 2, 1).to(device), sm_wl-400, ta_ss, ta_sd, ta_oo, ta_do, ra_so, ra_dd, T_g, return_toc=True)
                if print_flag:
                    print('now is using selected wavelength:')
                    print(sm_wl)
                    print_flag = False
                spec_data = (R_TOA)
                para_list.append(many_paras.cpu().numpy())

            if param_type=='prosail':

                tts_rad = torch.deg2rad(tts)

                sin_90tts = torch.sin(math.pi / 2 - tts_rad)

                skyl = 0.847 - 1.61 * sin_90tts + 1.04 * sin_90tts ** 2
                PARdiro = (1.0 - skyl)
                PARdifo = skyl

                #print(sin_90tts, skyl, PARdiro, PARdifo) for debug

                sin_90tts = torch.sin(math.pi / 2 - tts)

                skyl = 0.847 - 1.61 * sin_90tts + 1.04 * sin_90tts ** 2
                PARdiro = (1.0 - skyl)#.unsqueeze(1)
                PARdifo = skyl#.unsqueeze(1)
                resv = (spec_data[2].to(device) * PARdifo + spec_data[3].to(device) * PARdiro) / (PARdiro + PARdifo)
                spec_data = resv.T
                para_list.append(many_paras[:,:15].cpu().numpy())
        ref_list.append(spec_data.cpu().numpy())

    # Combine lists into arrays
    para_list = np.vstack(para_list)
    ref_list = np.vstack(ref_list)

    if std != 0:
        noise = np.random.normal(loc=0.0, scale=std, size=ref_list.shape)
        ref_list += noise

    if wavelength is not None and len(wavelength) > 0:
      return ref_list[:,wavelength-400], para_list#[:,wavelength-400]
    else:
      return ref_list, para_list