from flash.core.data.data_module import DataModule


class AudioSourceSeparationData(DataModule):
    """Data module for semantic segmentation tasks."""

    preprocess_cls = SemanticSegmentationPreprocess

    @staticmethod
    def configure_data_fetcher(
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> 'SegmentationMatplotlibVisualization':
        return SegmentationMatplotlibVisualization(labels_map=labels_map)

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @classmethod
    def from_data_source(
        cls,
        data_source: str,
        train_data: Any = None,
        val_data: Any = None,
        test_data: Any = None,
        predict_data: Any = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **preprocess_kwargs: Any,
    ) -> 'DataModule':

        if 'num_classes' not in preprocess_kwargs:
            raise MisconfigurationException("`num_classes` should be provided during instantiation.")

        num_classes = preprocess_kwargs["num_classes"]

        labels_map = getattr(preprocess_kwargs, "labels_map",
                             None) or SegmentationLabels.create_random_labels_map(num_classes)

        data_fetcher = data_fetcher or cls.configure_data_fetcher(labels_map)

        if flash._IS_TESTING:
            data_fetcher.block_viz_window = True

        dm = super(SemanticSegmentationData, cls).from_data_source(
            data_source=data_source,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            predict_data=predict_data,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            **preprocess_kwargs
        )

        if dm.train_dataset is not None:
            dm.train_dataset.num_classes = num_classes
        return dm

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        train_target_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_target_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_target_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        **preprocess_kwargs,
    ) -> 'DataModule':
        """Creates a :class:`~flash.image.segmentation.data.SemanticSegmentationData` object from the given data
        folders and corresponding target folders.
        Args:
            train_folder: The folder containing the train data.
            train_target_folder: The folder containing the train targets (targets must have the same file name as their
                corresponding inputs).
            val_folder: The folder containing the validation data.
            val_target_folder: The folder containing the validation targets (targets must have the same file name as
                their corresponding inputs).
            test_folder: The folder containing the test data.
            test_target_folder: The folder containing the test targets (targets must have the same file name as their
                corresponding inputs).
            predict_folder: The folder containing the predict data.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_classes: Number of classes within the segmentation mask.
            labels_map: Mapping between a class_id and its corresponding color.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.
        Returns:
            The constructed data module.
        Examples::
            data_module = SemanticSegmentationData.from_folders(
                train_folder="train_folder",
                train_target_folder="train_masks",
            )
        """
        return cls.from_data_source(
            DefaultDataSources.FOLDERS,
            (train_folder, train_target_folder),
            (val_folder, val_target_folder),
            (test_folder, test_target_folder),
            predict_folder,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            labels_map=labels_map,
            **preprocess_kwargs,
        )

        
        
        
        
        
# test
datamodule = AudioSourceSeparationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
)
