# Unit test initialize_window_features
# ==============================================================================
import re
import pytest
from skforecast.utils import initialize_window_features


class WindowFeature:
    def __init__(self, window_sizes, features_names):
        self.window_sizes = window_sizes
        self.features_names = features_names

    def transform_batch(self):
        pass

    def transform(self):
        pass


class InvalidWindowFeatureNoAttributes:
    def transform_batch(self):
        pass

    def transform(self):
        pass


class InvalidWindowFeatureNoMethods:
    def __init__(self, window_sizes, features_names):
        self.window_sizes = window_sizes
        self.features_names = features_names


def test_ValueError_initialize_window_features_when_empty_list():
    """
    Test ValueError is raised when `window_features` is an empty list.
    """
    err_msg = re.escape(
        "Argument `window_features` must contain at least one element."
    )
    with pytest.raises(ValueError, match = err_msg):
        initialize_window_features(window_features=[])


def test_ValueError_initialize_window_features_when_no_required_attributes():
    """
    Test ValueError is raised when window feature does not have the required attributes.
    """
    err_msg = re.escape(
        ("InvalidWindowFeatureNoAttributes must have the attributes: "
         "['window_sizes', 'features_names'].")
    )
    with pytest.raises(ValueError, match = err_msg):
        initialize_window_features(InvalidWindowFeatureNoAttributes())


def test_ValueError_initialize_window_features_when_no_required_methods():
    """
    Test ValueError is raised when window feature does not have the required attributes.
    """
    err_msg = re.escape(
        ("InvalidWindowFeatureNoMethods must have the methods: "
         "['transform_batch', 'transform'].")
    )
    with pytest.raises(ValueError, match = err_msg):
        initialize_window_features(InvalidWindowFeatureNoMethods(5, 'feature_1'))


def test_TypeError_initialize_window_features_when_window_sizes_not_int_list():
    """
    Test TypeError is raised when `window_sizes` is not an int or a list of ints.
    """
    window_sizes = 'not_valid'

    err_msg = re.escape(
        (f"Attribute `window_sizes` of WindowFeature must be an int or a list "
         f"of ints. Got {type(window_sizes)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        initialize_window_features(WindowFeature(window_sizes, 'feature_1'))


def test_ValueError_initialize_window_features_when_window_sizes_int_lower_than_1():
    """
    Test ValueError is raised when `window_sizes` is an int lower than 1.
    """
    err_msg = re.escape(
        ("If argument `window_sizes` is an integer, it must be equal "
         "to or greater than 1. Got 0 from WindowFeature.")
    )
    with pytest.raises(ValueError, match = err_msg):
        initialize_window_features(WindowFeature(0, 'feature_1'))


@pytest.mark.parametrize("window_sizes", 
                         [[1, 2, 1.], [1, 3, 0], ['1', 3, 5]], 
                         ids = lambda window_sizes: f'window_sizes: {window_sizes}')
def test_ValueError_initialize_window_features_when_window_sizes_list_not_int_or_lower_than_1(window_sizes):
    """
    Test ValueError is raised when `window_sizes` is a list with elements not 
    int or lower than 1.
    """
    err_msg = re.escape(
        (f"If argument `window_sizes` is a list, all elements must be integers "
         f"equal to or greater than 1. Got {window_sizes} from WindowFeature.")
    )
    with pytest.raises(ValueError, match = err_msg):
        initialize_window_features(WindowFeature(window_sizes, 'feature_1'))


def test_TypeError_initialize_window_features_when_features_names_not_str_list():
    """
    Test TypeError is raised when `features_names` is not a str or a list of strs.
    """
    features_names = 5

    err_msg = re.escape(
        (f"Attribute `features_names` of WindowFeature must be a str or a list "
         f"of strings. Got {type(features_names)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        initialize_window_features(WindowFeature(3, features_names))


def test_TypeError_initialize_window_features_when_features_names_list_not_str():
    """
    Test TypeError is raised when `features_names` is a list with elements not str.
    """
    features_names = ['name_1', 5]

    err_msg = re.escape(
        ("If argument `features_names` is a list, all elements "
         "must be strings. Got ['name_1', 5] from WindowFeature.")
    )
    with pytest.raises(TypeError, match = err_msg):
        initialize_window_features(WindowFeature(3, features_names))


def test_ValueError_initialize_window_features_when_not_unique_features_names():
    """
    Test ValueError is raised when `features_names` are not unique.
    """
    window_features = [
        WindowFeature([4, 6], ['feature_1', 'feature_2']),
        WindowFeature(5, ['feature_2', 'feature_3'])
    ]
    window_features_names = ['feature_1', 'feature_2', 'feature_2', 'feature_3']

    err_msg = re.escape(
        (f"All window features names must be unique. Got {window_features_names}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        initialize_window_features(window_features)


def test_initialize_window_features_when_None():
    """
    Test initialize_window_features with None.
    """

    window_features, window_features_names, max_size_window_features = (
        initialize_window_features(None)
    )

    assert window_features is None
    assert window_features_names is None
    assert max_size_window_features is None


def test_initialize_window_features_valid():
    """
    Test initialize_window_features with valid values.
    """

    wf1 = WindowFeature(5, "feature1")
    wf2 = WindowFeature([3, 4], ["feature2", "feature3"])
    window_features, window_features_names, max_size_window_features = (
        initialize_window_features([wf1, wf2])
    )

    assert window_features == [wf1, wf2]
    assert window_features_names == ["feature1", "feature2", "feature3"]
    assert max_size_window_features == 5
