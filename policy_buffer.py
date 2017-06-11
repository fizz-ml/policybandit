class PolicyBuffer:
    """
    Experience Replay stores action, state, reward and terminal signal
    for each time step.
    """
    def __init__(self, capacity = None):
        """ Creates a policy bueffer of certain capacity.
            Acts like a circular buffer.
        Args:
            capacity:       The capacity of the policy replay buffer.
                            If None the buffer will have unlimited size.
        """
        self.capacity = capacity
        self._buffer = []
        self.current_index = 0
        self.length = 0

    def put(self, policy_dict, ave_reward):
        ''' Adds a new policy tuple
        '''
        pol_tuple = (policy_dict, ave_reward)
        if self.length < self.capacity or not capacity:
            self._buffer.append( pol_tuple )
        else:
            self._buffer[self.current_index] = pol_tuple
        self._icrement_index()

    def peek(self):
        ''' Returns the most recent policy tuple
        '''
        return self._buffer[self.current_index]

    def _increment_index(self):
        if self.capacity:
            self.current_index = (self.current_index + 1) % self.capacity
            self.length = min(self.capacity, self.length + 1)
        else:
            self.current_index += 1
            self.length += 1

    def __iter__(self):
        return iter(self._buffer)

