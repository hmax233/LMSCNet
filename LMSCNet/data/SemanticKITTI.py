from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np
import yaml
import random
import sys
#yaml中的数据按数据前空格进行分级，第一级的和其之下的构成字典
import LMSCNet.data.io_data as SemanticKittiIO


class SemanticKITTI_dataloader(Dataset):

  def __init__(self, dataset, phase):
    '''

    :param dataset: The dataset configuration (data augmentation, input encoding, etc)
    :param phase_tag: To differentiate between training, validation and test phase
    '''

    yaml_path, _ = os.path.split(os.path.realpath(__file__))
    self.dataset_config = yaml.safe_load(open(os.path.join(yaml_path, 'semantic-kitti.yaml'), 'r'))#载入yaml格式的config文件，应该是以字典格式读入
    self.nbr_classes = self.dataset_config['nbr_classes']
    self.grid_dimensions = self.dataset_config['grid_dims']   # [W, H, D]
    self.remap_lut = self.get_remap_lut()#重映射到附近区域
    self.rgb_mean = np.array([0.34749558, 0.36745213, 0.36123651])  # images mean:  [88.61137282 93.70029365 92.11530949]
    self.rgb_std = np.array([0.30599035, 0.3129534 , 0.31933814])   # images std:  [78.02753826 79.80311686 81.43122464]
    #统计输入的mean和std用于normalization
    self.root_dir = dataset['ROOT_DIR']
    #modalities分为3D_LABEL, 3D_OCCUPANCY, 3D_OCCLUDED 3D_INVALID.
    self.modalities = dataset['MODALITIES']
    self.extensions = {'3D_OCCUPANCY': '.bin', '3D_LABEL': '.label', '3D_OCCLUDED': '.occluded',
                       '3D_INVALID': '.invalid'}
    self.data_augmentation = {'FLIPS': dataset['AUGMENTATION']['FLIPS']}#进行图像增强，推断为flips

    self.filepaths = {}
    #filepaths为一个dict,dict中的value也为dict
    #dict中的key分别为"3D_LABEL" "3D_OCCUPANCY"  "3D_OCCLUDED"
    self.phase = phase
    self.class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                       6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                       2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                       2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                       2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])
   #统计各类的出现频率，用于之后的loss_weight
  
    self.split = {'train': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10], 'val': [8],
                  'test': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]}

    for modality in self.modalities:
      if self.modalities[modality]:
        self.get_filepaths(modality)

    # if self.phase != 'test':
    #   self.check_same_nbr_files()

    self.nbr_files = len(self.filepaths['3D_OCCUPANCY'])  # TODO: Pass to something generic

    return

  def get_filepaths(self, modality):
    '''
    Set modality filepaths with split according to phase (train, val, test)
    '''

    sequences = list(sorted(glob(os.path.join(self.root_dir, 'dataset', 'sequences', '*')))[i] for i in self.split[self.phase])

    if self.phase != 'test':
    #为何对modality进行分类，分为了3D_LABEL, 3D_OCCLUDED, 3D_OCCUPANCY
    #分别读取不同的数据
      if modality == '3D_LABEL':
        self.filepaths['3D_LABEL'] = {'1_1': [], '1_2': [], '1_4': [], '1_8': []}
        self.filepaths['3D_INVALID'] = {'1_1': [], '1_2': [], '1_4': [], '1_8': []}
        for sequence in sequences:
          assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
          # Scale 1:1
          self.filepaths['3D_LABEL']['1_1'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label')))
          self.filepaths['3D_INVALID']['1_1'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid')))
          # Scale 1:2
          self.filepaths['3D_LABEL']['1_2'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label_1_2')))
          self.filepaths['3D_INVALID']['1_2'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid_1_2')))
          # Scale 1:4
          self.filepaths['3D_LABEL']['1_4'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label_1_4')))
          self.filepaths['3D_INVALID']['1_4'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid_1_4')))
          # Scale 1:8
          self.filepaths['3D_LABEL']['1_8'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label_1_8')))
          self.filepaths['3D_INVALID']['1_8'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid_1_8')))

      if modality == '3D_OCCLUDED':
        self.filepaths['3D_OCCLUDED'] = []
        for sequence in sequences:
          assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
          self.filepaths['3D_OCCLUDED'] += sorted(glob(os.path.join(sequence, 'voxels', '*.occluded')))
    
    if modality == '3D_OCCUPANCY':
      self.filepaths['3D_OCCUPANCY'] = []
      for sequence in sequences:
        assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
        self.filepaths['3D_OCCUPANCY'] += sorted(glob(os.path.join(sequence, 'voxels', '*.bin')))

    # if modality == '2D_RGB':
    #   self.filepaths['2D_RGB'] = []
    #   for sequence in sequences:
    #     assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
    #     self.filepaths['2D_RGB'] += sorted(glob(os.path.join(sequence, 'image_2', '*.png')))[::5]

    return

  def check_same_nbr_files(self):
    '''
    Set modality filepaths with split according to phase (train, val, test)
    '''

    # TODO: Modify for nested dictionaries...
    for i in range(len(self.filepaths.keys()) - 1):
      length1 = len(self.filepaths[list(self.filepaths.keys())[i]])
      length2 = len(self.filepaths[list(self.filepaths.keys())[i+1]])
      assert length1 == length2, 'Error: {} and {} not same number of files'.format(list(self.filepaths.keys())[i],
                                                                                    list(self.filepaths.keys())[i+1])
    return

  def __getitem__(self, idx):
    '''

    '''

    data = {}

    do_flip = 0
    if self.data_augmentation['FLIPS'] and self.phase == 'train':
      do_flip = random.randint(0, 3)

    for modality in self.modalities:
      if (self.modalities[modality]) and (modality in self.filepaths):
        data[modality] = self.get_data_modality(modality, idx, do_flip)

    return data, idx

  def get_data_modality(self, modality, idx, flip):

    if modality == '3D_OCCUPANCY':
      
      OCCUPANCY = SemanticKittiIO._read_occupancy_SemKITTI(self.filepaths[modality][idx])
      OCCUPANCY = np.moveaxis(OCCUPANCY.reshape([self.grid_dimensions[0],
                                                 self.grid_dimensions[2],
                                                 self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])
      OCCUPANCY = SemanticKittiIO.data_augmentation_3Dflips(flip, OCCUPANCY)
      return OCCUPANCY[None, :, :, :]

    elif modality == '3D_LABEL':
      LABEL_1_1 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_1', idx))
      LABEL_1_2 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_2', idx))
      LABEL_1_4 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_4', idx))
      LABEL_1_8 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_8', idx))
      return {'1_1': LABEL_1_1, '1_2': LABEL_1_2, '1_4': LABEL_1_4, '1_8': LABEL_1_8}

    elif modality == '3D_OCCLUDED':
      OCCLUDED = SemanticKittiIO._read_occluded_SemKITTI(self.filepaths[modality][idx])
      OCCLUDED = np.moveaxis(OCCLUDED.reshape([self.grid_dimensions[0],
                                               self.grid_dimensions[2],
                                               self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])
      OCCLUDED = SemanticKittiIO.data_augmentation_3Dflips(flip, OCCLUDED)
      return OCCLUDED

    # elif modality == '2D_RGB':
    #   RGB = SemanticKittiIO._read_rgb_SemKITTI(self.filepaths[modality][idx])
    #   # TODO Standarize, Normalize
    #   RGB = SemanticKittiIO.img_normalize(RGB, self.rgb_mean, self.rgb_std)
    #   RGB = np.moveaxis(RGB, (0, 1, 2), (1, 2, 0)).astype(dtype='float32')  # reshaping [3xHxW]
    #   # There is a problem on the RGB images.. They are not all the same size and I used those to calculate the mapping
    #   # for the sketch... I need images all te same size..
    #   return RGB

    else:
      assert False, 'Specified modality not found'

  def get_label_at_scale(self, scale, idx):

    scale_divide = int(scale[-1])
    INVALID = SemanticKittiIO._read_invalid_SemKITTI(self.filepaths['3D_INVALID'][scale][idx])
    LABEL = SemanticKittiIO._read_label_SemKITTI(self.filepaths['3D_LABEL'][scale][idx])
    if scale == '1_1':
      LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC remap再交换
    LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
    LABEL = np.moveaxis(LABEL.reshape([int(self.grid_dimensions[0] / scale_divide),
                                       int(self.grid_dimensions[2] / scale_divide),
                                       int(self.grid_dimensions[1] / scale_divide)]), [0, 1, 2], [0, 2, 1])

    return LABEL

  def read_semantics_config(self, data_path):

    # get number of interest classes, and the label mappings
    DATA = yaml.safe_load(open(data_path, 'r'))
    self.class_strings = DATA["labels"]
    self.class_remap = DATA["learning_map"]
    self.class_inv_remap = DATA["learning_map_inv"]
    self.class_ignore = DATA["learning_ignore"]
    self.n_classes = len(self.class_inv_remap)

    return

  def get_inv_remap_lut(self):
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''

    # make lookup table for mapping
    maxkey = max(self.dataset_config['learning_map_inv'].keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
    remap_lut[list(self.dataset_config['learning_map_inv'].keys())] = list(self.dataset_config['learning_map_inv'].values())

    return remap_lut

  def get_remap_lut(self):
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''

    # make lookup table for mapping为映射制作查找表
    
    #learning_map参照semantic-kiiti.yaml 
    #learning_map左边为34类，右边为20类，两者之间相互对应,应该都是数据集提供的标签，左边的标签多一点是因为又进一步的分了moving和non-moving类
    
    
    maxkey = max(self.dataset_config['learning_map'].keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(self.dataset_config['learning_map'].keys())] = list(self.dataset_config['learning_map'].values())
    
    # in completion we have to distinguish empty and invalid voxels.
    # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
    remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
    remap_lut[0] = 0  # only 'empty' stays 'empty'.本来0为255，这里将0再次设置为0，0应该是作为empty
#破案了，这里的255和后面损失函数中的indice还是index相对应
    return remap_lut

  def __len__(self):
    """
    Returns the length of the dataset
    """
    # Return the number of elements in the dataset
    return self.nbr_files

