class Agent:
    """The generic interface for an agent.
    """

    def __init__(self):
        pass

    def train():
        """Trains the agent for a bit.

            Args:
                None
            Returns:
                None
        """
        raise NotImplementedError

    def get_next_action(cur_state,
            agent_id=None,
            is_test=True):
        """Get the next action from the agent.

            Takes a state and returns the next action from the agent.

            Args:
                cur_state: The current state of the enviroment
            Returns:
                The next action that the agent will carry out given the current state
        """
        raise NotImplementedError

    def save_model(location=None):
        """Save the model to a given location

            Args:
                Location: where to save the model
            Returns:
                None
        """
        raise NotImplementedError

    def load_model(location=None):
        """Loads the models from a given location

            Args:
                Location: from where to load the model
            Returns:
                None
        """
        raise NotImplementedError
