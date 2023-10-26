import pygrenade_vx as grenade


class EntityOnExecutionInstance:
    """
    Mixin of execution instance property for use in modules.
    """

    def __init__(
            self,
            execution_instance: grenade.common.ExecutionInstanceID) -> None:
        """
        :param execution_instance: Execution instance to place to.
        """
        self.execution_instance = execution_instance
