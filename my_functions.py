from adasim_utils.dataset import DatasetFolderAdaSim
import torch
import os


def make_index_label(args, transform):
        #Make index_label:
    dataset_graph = DatasetFolderAdaSim(args.data_path, args, transform=transform, return_index_instead_of_target=False)
    sampler_graph = torch.utils.data.DistributedSampler(dataset_graph)

    data_loader_graph = torch.utils.data.DataLoader(
        dataset_graph,
        sampler=sampler_graph,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded: there are {len(dataset_graph)} images.")
    

    # Filename to check and write to
    filename = f"index_label_image_{os.path.split(args.untar_path)[1]}.txt"

    # Check if the file exists
    if not os.path.exists(filename):
        # If the file does not exist, execute the code and write the results
        index_label_image = [None] * len(dataset_graph)

        for it, (image, label, same_im, neighbors, index, image_name_list) in enumerate(data_loader_graph):
            index_list = index.tolist()
            label_list = label.tolist()
            for index, label, image_name in zip(index_list, label_list, image_name_list):
                index_label_image[index] = str(label)# + '_' + image_name

        # Write the results to the file
        with open(filename, 'w') as file:
            for item in index_label_image:
                file.write("%s\n" % item)
    else:
        print(f"The file '{filename}' already exists.")
        with open(filename, 'r') as file:
            index_label_image = [line.strip() for line in file]

    index_label_int = list(map(int, index_label_image))
    return index_label_int