class SpookyBatch:
    device = torch.device('cuda')

    def __init__(self):
        self.N = 0
        self.Z = []
        self.R = []
        self.E = []
        self.idx_i = []
        self.idx_j = []
        self.batch_seg = []

    def toTensor(self):
        self.Z = torch.tensor(self.Z,dtype=torch.int64,device=SpookyBatch.device)
        self.R = torch.tensor(self.R,dtype=torch.float32,device=SpookyBatch.device,requires_grad=True)
        self.Q = torch.zeros(self.N,dtype=torch.float32,device=SpookyBatch.device) # not using this so could just pass the same tensor around
        self.S = torch.zeros(self.N,dtype=torch.float32,device=SpookyBatch.device) # ditto
        self.E = torch.tensor(self.E,dtype=torch.float32,device=SpookyBatch.device)
        self.idx_i = torch.tensor(self.idx_i,dtype=torch.int64,device=SpookyBatch.device)
        self.idx_j = torch.tensor(self.idx_j,dtype=torch.int64,device=SpookyBatch.device)
        self.batch_seg = torch.tensor(self.batch_seg,dtype=torch.int64,device=SpookyBatch.device) # int64 required for "index tensors"
        return self

def load_batches(my_mols): # my_mols == some structure which has your loaded mol data, prob retrieved from a file,
                                              # or you can load it from a file here on demand to save memory
    batches = []
    batch = None
    nm = 0 # how many mols in current batch
    NM = 100 # how many mols we put in each batch
    for m in my_mols: 
        if nm == 0:
            na = 0 # num total atoms in this batch
            batch = SpookyBatch() # stores the data in a format we can pass to SpookyNet
        
        batch.Z.extend(m.species)
        batch.R.extend(m.coords)
        batch.E.append(m.energy) # target energy
        cur_idx_i,cur_idx_j = get_idx(m.coords) # see below but also look at SpookyNetCalculator for more options
        cur_idx_i += na
        cur_idx_j += na
        batch.idx_i.extend(cur_idx_i)
        batch.idx_j.extend(cur_idx_j)
        batch.batch_seg.extend([nm]*len(m.species))
        na += len(m.species)
        nm += 1

        if nm >= NM:
            batch.N = nm
            batches.append(batch.toTensor()) # or you could convert to a tensor during training, depends on how much memory you have
            nm = 0 
    if batch:
        batches.append(batch.toTensor())
    return batches

# taken from SpookyNetCalculator 
def get_idx(R):
    N = len(R)
    idx = torch.arange(N,dtype=torch.int64)
    idx_i = idx.view(-1, 1).expand(-1, N).reshape(-1)
    idx_j = idx.view(1, -1).expand(N, -1).reshape(-1)
    # exclude self-interactions
    nidx_i = idx_i[idx_i != idx_j]
    nidx_j = idx_j[idx_i != idx_j]
    return nidx_i.numpy(),nidx_j.numpy() # kind of dumb converting to numpy when we use torch later, but it fits our model

def train():
    NUM_EPOCHES = 1000
    BEST_POINT = 'best.pt'

    model = SpookyNet().to(torch.float32).cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(),lr=START_LR,amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25, threshold=0)


    training = load_batches(my_training_mols)
    validation = load_batches(my_validation_mols)
    mse_sum = torch.nn.MSELoss(reduction='sum')

    for epoch in range(NUM_EPOCHES):
        random.shuffle(training)

        for batch in training:
            N = batch.N
            res = model.energy(Z=batch.Z,Q=batch.Q,S=batch.S,R=batch.R,idx_i=batch.idx_i,idx_j=batch.idx_j,batch_seg=batch.batch_seg,num_batch=N)
            E = res[0]
     
            loss = mse_sum(E, batch.E)/N

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        rmse = compute_rmse(validation,model)
        if scheduler.is_better(rmse, scheduler.best):
            model.save(BEST_POINT)

        scheduler.step(rmse)
        print('Epoch: {} / LR: {} / RMSE: {:.3f} / Best: {:.3f}'.format(scheduler.last_epoch,learning_rate,rmse,scheduler.best))

def compute_rmse(batches,model):
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    model.eval()
    for batch in batches:
        N = batch.N
        res = model.energy(Z=batch.Z,Q=batch.Q,S=batch.S,R=batch.R,idx_i=batch.idx_i,idx_j=batch.idx_j,batch_seg=batch.batch_seg,num_batch=N)
        E = res[0]

        # sum over pairings
        total_mse += mse_sum(E, batch.E).item()
        count += N

    model.train()
    return math.sqrt(total_mse / count)