class PipelineState():

    def __init__(
                self, 
                initial_pipeline_state: Dict[str, PipelineDataContainer] = None, 
                add_dummy_data: bool = False
                ) -> None:
        if initial_pipeline_state is None:
            initial_pipeline_state = {}
        self.__pipelines: Dict[str, PipelineDataContainer] = initial_pipeline_state

        if add_dummy_data is True:
            self.__init_with_dummy_data()