import torch

num_classes = 100 # for ImageNet100, or 10 for Cifar10
num_train = 130000 # for ImageNet100, or 50000 for Cifar10
num_val = 5000 # for ImageNet100, or 10000 for Cifar10
num_total = num_train + num_val

num_samples_per_class_train = num_train // num_classes
num_samples_per_class_val = num_val // num_classes

train_mask = torch.zeros(num_total, dtype=torch.bool)
train_mask[:num_train] = True
val_mask = ~train_mask

y_train = torch.zeros(num_train, dtype=torch.long)
y_val = torch.zeros(num_val, dtype=torch.long)

for i in range(num_classes):
    train_indices = torch.arange(i*num_samples_per_class_train, (i+1)*num_samples_per_class_train)
    val_indices = torch.arange(i*num_samples_per_class_val, (i+1)*num_samples_per_class_val)
    y_train[train_indices] = i
    y_val[val_indices] = i

# print(y_train)
# print(y_val)
y = torch.cat([y_train, y_val], dim=0)
print(y)


teacher_graph.num_nodes = num_total
teacher_graph.edge_index = torch.cat([teacher_graph.edge_index, teacher_graph_test.edge_index], dim=1)
teacher_graph.x = torch.cat([teacher_graph.x, teacher_graph_test.x], dim=0)
teacher_graph.train_mask = train_mask
teacher_graph.val_mask = val_mask
teacher_graph.y = y

student_graph.num_nodes = num_total
student_graph.edge_index = torch.cat([student_graph.edge_index, student_graph_test.edge_index], dim=1)
student_graph.x = torch.cat([student_graph.x, student_graph_test.x], dim=0)
student_graph.train_mask = train_mask
student_graph.val_mask = val_mask
student_graph.y = y

torch.save({'teacher_graph':teacher_graph, 'student_graph':student_graph}, '/home/alij/SSL-GraphNNCLR/saved_graphs/ImageNet100/ImageNet100_graphs_epn15_vitb8.pth')

