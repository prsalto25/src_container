class EmptyVA:
    def __init__(self, **kwargs):  # Corrected kwargs syntax
        self.outstream = kwargs.get('outstream', None)  # Extract outstream safely
    
    def run(self, frame):
        print("going here inside va")
        if self.outstream:  # Check if outstream is not None
            self.outstream.write(frame)
        else:
            print("Error: outstream is not defined!")
