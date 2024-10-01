import os
from data.transform import get_data, Totensor, MyDataset, destandardize_norm
from torch.utils.data import DataLoader
from models.MFDNN import *
from functions.functions import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## get data
xl, yl, xh, yh, xh_test, yh_test, mean, std = get_data()
xl_tensor, yl_tensor, xh_tensor, yh_tensor, xh_test_tensor, yh_test_tensor = Totensor(xl, yl, xh, yh, xh_test, yh_test, device)


# train Lmodel
lmodel = LModel().to(device)
lModel_optimizer = torch.optim.Adam(params=lmodel.parameters(), lr=0.001)
lmodel, lModel_loss = trainLModel(lmodel, lModel_optimizer, xl_tensor, yl_tensor, 500000, "LModel Train")
torch.save(lmodel.state_dict(), os.path.join('results/model_saved', "Lmodel.pth"))

# lmodel = LModel().to(device)
# lmodel.load_state_dict(torch.load('results/model_saved/Lmodel.pth'))

xh_yl = lmodel(xh_tensor)

model = MFDNN2().to(device)

# train model
AtMFDNN_train_model(model, xl_tensor, yl_tensor, xh_tensor, xh_yl, yh_tensor, 'results/model_saved')
train_model(model, xl_tensor, yl_tensor, xh_tensor, yh_tensor, 'results/model_saved')

# test model
model = load_model(model, 'results/model_saved/AtMFDNN_final.pth')
_, pred_yh_test = model(xh_test_tensor, xh_test_tensor)
test_loss = F.mse_loss(pred_yh_test, yh_test_tensor)

# show result


pred_yh_test = pred_yh_test.cpu().detach().numpy()
xh_test_tensor = xh_test_tensor.cpu().detach().numpy()
pred = np.concatenate((xh_test_tensor, pred_yh_test), axis=1)
pred = destandardize_norm(pred, mean, std)

show_result(pred)
