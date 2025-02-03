import secretflow as sf


class Party(object):
    def __init__(self, args, party: sf.SPU):
        self.args = args
        self.party = party
