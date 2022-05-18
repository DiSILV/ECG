if not(LoadSavedModels):
    #---------- Model training----------
    #losses
    reconstruction_criterion = GetReconstructionLossFun(hp.AeLoss)
    independence_criterion = GetIndependenceLossFun(hp.IndLoss)

    #model
    input_norm_fun = GetInputNormalizationFunction(hp.NormInput)
    enc, dec = GetAE(hp.ModelType, ActivationToUse=hp.ActivationToUse, nChannels=hp.nChannelsAb, Dim=hp.ModelDim,UseBN=hp.UseBN, UseSN=hp.UseSN)
    disc_enc, _ = GetAE(hp.ModelType, ActivationToUse=hp.ActivationToUse, nChannels=hp.nChannelsCh, Dim=hp.ModelDim,UseBN=hp.UseBN, UseSN=hp.UseSN)
    modules_list = [enc, dec, disc_enc]
    if hp.IndLoss=='Pearson':
        X_disc = PearsonDisc(z_dim=hp.ModelDim)
        T_disc = PearsonDisc(z_dim=hp.ModelDim)
        modules_list.extend([X_disc,T_disc])
    [x.to(device) for x in  modules_list]
    [x.train() for x in modules_list]

    # optimizers
    ae_parameters = list(set(list(enc.parameters()) + list(dec.parameters()) + list(disc_enc.parameters())))
    disc_parameters = list(disc_enc.parameters())
    if hp.IndLoss == 'Pearson':
        disc_parameters.extend(list(X_disc.parameters())+ list(T_disc.parameters()))
    ae_optimizer = optim.Adam(ae_parameters, lr=hp.LearningRateAE, weight_decay=hp.WeightDecay)
    disc_optimizer = optim.Adam(disc_parameters, lr=hp.LearningRateD, weight_decay=hp.WeightDecay)
    if hp.UseScheduler:
        ae_optimizer = optim.SGD(ae_parameters, lr=hp.LearningRateAE, weight_decay=hp.WeightDecay)
        ae_scheduler = MultiStepLR(ae_optimizer, milestones=milestones, gamma=0.1)

    #logger
    # if hp.LogToWandb:
    #     wandb.login()
    #     wandb.init(project=hp.ProjectName, entity="orkatz")
    #     wandb.run.name = hp.ExpName
    #     # wandb.watch(enc, disc_enc, log="all")
    #     config = wandb.config
    #     config.args = vars(hp)

    # print('----------------Training-----------------')
    AELossVec, DiscLossVec = [], []
    # hp.NumberOfBatches=10000
    pbar = tqdm(range(hp.NumberOfBatches))
    for b in pbar:
        ae_scheduler.step() if hp.UseScheduler else None
        X, T, D, MHR, FHR, _ = data_loader.GetBatch(BatchSize=hp.BatchSize, seg_len=hp.InputSegment)
        [X, T] = [input_norm_fun(x) for x in [X, T]]

        disc_enc.zero_grad()
        x_code = enc(X)
        t_code_disc = disc_enc(T)
        independence_loss = independence_criterion(x_code, t_code_disc)

        independence_loss.backward()
        disc_optimizer.step()
        DiscLossVec.append(independence_loss.item())

        if b % hp.AeFac == 0:
            enc.zero_grad()
            disc_enc.zero_grad()
            dec.zero_grad()

            x_code = enc(X)
            t_code_disc = disc_enc(T)
            x_dec = dec(x_code, T.permute((0,2,1)))
            reconstruction_loss_ae = reconstruction_criterion(x_dec, X)
            independence_loss_ae = independence_criterion(x_code, t_code_disc)
            loss_ae = reconstruction_loss_ae - independence_loss_ae * hp.Beta

            loss_ae.backward()
            ae_optimizer.step()
            AELossVec.append(loss_ae.item())

        # log values to console
        if b > hp.AeFac:
            pbar.set_postfix({'Exp': ie, 'ae-loss': np.log10(reconstruction_loss_ae.item()),
                              'disc-loss': np.log10(np.abs(independence_loss_ae.item()))})
        # log values to console
        if hp.LogToWandb:
            wandb.log({
                'AE loss': np.log10(reconstruction_loss_ae.item()),
                'Discriminator loss': np.log10(np.abs(independence_loss_ae.item()))
            })
    torch.save(enc.state_dict(), os.path.join(os.getcwd(),ModelsFolder,'Sub%g_Encoder'%hp.Subjects[0]))
else:
    input_norm_fun = GetInputNormalizationFunction(hp.NormInput)
    enc, _ = GetAE(hp.ModelType, ActivationToUse=hp.ActivationToUse, nChannels=hp.nChannelsAb, Dim=hp.ModelDim,UseBN=hp.UseBN, UseSN=hp.UseSN)
    enc.load_state_dict(torch.load(os.path.join(os.getcwd(),ModelsFolder,'Sub%g_Encoder'%hp.Subjects[0]),map_location="cpu"))
    enc.eval()
    # hp.LogToWandb=False

enc.to('cpu')