#!/home/miranda9/.conda/envs/automl-meta-learning/bin/python3.7
#SBATCH --job-name="miranda9job"
#SBATCH --output="experiment_output_job.%j.%N.out"
#SBATCH --error="experiment_output_job.%j.%N.err"
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --mail-user=brando.science@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=04:00:00
#SBATCH --partition=secondary-Eth

import torch


out_features = 5

# resnet 18 (for single gpu test)
net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
net.fc = torch.nn.Linear(in_features=512, out_features=out_features, bias=True)

# resnet 152 (for multi gpu test)
# net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
# net.fc = torch.nn.Linear(in_features=2048, out_features=out_features, bias=True)
if torch.cuda.is_available():
    net = net.cuda()

print(type(net))

print(torch.cuda.device_count())
print(list(range(torch.cuda.device_count())))
if torch.cuda.device_count() > 1:
    # args.base_model = torch.nn.parallel.DistributedDataParallel(args.base_model, device_ids=list(range(torch.cuda.device_count())))
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()

print(type(net))

batch_size = 1024
x = torch.randn(batch_size, 3, 84, 84).cuda()
y_pred = net(x)
print(y_pred.size())
y = torch.randn(batch_size, out_features).cuda()

print(y_pred.sum())

criterion = torch.nn.MSELoss()

loss = criterion(y_pred, y)
print(loss)

print('DONE')

