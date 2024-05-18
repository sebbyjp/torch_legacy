import h5py
import numpy as np
import webdataset as wds
from oxe_torch.data.webdataset import from_hdf5, process_group
import unittest
import os

class TestWebDataset(unittest.TestCase):
    def setUp(self):
        # Create a fake HDF5 file with known data
        self.input_file = 'fake_input.hdf5'
        with h5py.File(self.input_file, 'w') as f:
            grp = f.create_group('entry1.npz')
            grp.create_dataset('data.npz', data=np.array([1, 2, 3]))
            dt = h5py.string_dtype(encoding='utf-8')
            grp.create_dataset('label.npz', data=np.array('label1', dtype=dt))

            grp = f.create_group('entry2.npz')
            grp.create_dataset('data', data=np.array([4, 5, 6]))
            grp.create_dataset('label', data=np.array('label2', dtype=dt))

        self.output_file = 'output'

    def test_from_hdf5(self):
        # Run the function
        try:
            from_hdf5(self.input_file, self.output_file)
        except Exception as e:
            self.fail(f'from_hdf5 raised an exception: {e}')

        # Check if the output file is created
        self.assertTrue(os.path.exists(f'{self.output_file}-000000.tar'))

        # Check if the output file is a valid WebDataset
        try:
            for i, sample in enumerate(wds.WebDataset(f'{self.output_file}-000000.tar')):
                # Check if the output file contains the expected data
                self.assertTrue(np.array_equal(sample['data.npy'], np.array([1, 2, 3]) if i == 0 else np.array([4, 5, 6])))
                self.assertEqual(sample['label.txt'].decode('utf-8'), 'label1' if i == 0 else 'label2')
        except Exception as e:
            self.fail(f'Output file is not a valid WebDataset or does not contain the expected data: {e}')

    def tearDown(self):
        # Clean up the created files
        os.remove(self.input_file)
        os.remove(f'{self.output_file}-000000.tar')

if __name__ == '__main__':
    unittest.main()