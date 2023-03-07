
from DatasetBuilder import DatasetBuilder
import DiffusionModelStructure
from GaussianDiffusion import GaussianDiffusion
from DiffusionModel import DiffusionModel

# Constants
# Values chosen in the paper
beta_start = 0.0001
beta_end = 0.02
time_steps = 1000
clip_min = -1.0
clip_max = 1.0

img_size = 64
img_channels = 3
batch_size =32

filters = 64
learning_rate = 0.0002

faces_directory_path = "C:/Users/allan/Downloads/GANFacesDateset"

dataset_builder = DatasetBuilder(directory_path=faces_directory_path, img_size=img_size, batch_size=batch_size)
dataset_builder.build_dataset()
dataset = dataset_builder.get_dataset()
network = DiffusionModelStructure.diff_model_structure(img_size=img_size, img_channels=img_channels, filters=filters)
network.summary()
gdf_util = GaussianDiffusion(beta_start=beta_start, beta_end=beta_end, time_steps=time_steps)
diffusion_model = DiffusionModel(
    network=network,
    time_steps=time_steps,
    gdf_util=gdf_util,
    img_size=img_size,
    img_channels=img_channels,
    learning_rate=learning_rate,
    clip_min=clip_min,
    clip_max=clip_max)
diffusion_model.train_model(dataset, epochs=500)
