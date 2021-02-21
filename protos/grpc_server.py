class LearnerServer(learner_pb2_grpc.LearnerServiceServicer):

    def __init__(self):
        self.model = Model()
        self.queue = Queue()

        self.learner = Learner(self.model, self.queue, daemon=True)
        self.learner.start()

        self.deployer = Deployer(self.model, daemon=True)
        self.deployer.start()

    def Enqueue(self, request, context):
        assert type(request) is learner_pb2.History
        print('Enqueue: History(inputs=%s, output=%s)' % (request.inputs[0], request.output[0]))

        self.queue.put(request)

        return Empty()

    def Deploy(self, request, context):
        print('LearnerServer.Deploy')
        bytes_ = b''
        with open(Deployer.PATH, 'rb') as f:
            bytes_ = f.read()
        print('Chunk size: %d (%d blocks)' % (len(bytes_), len(bytes_) // GRPC_CHUNK_SIZE + 1))
        while bytes_:
            data = bytes_[:GRPC_CHUNK_SIZE]
            try:
                chunk = learner_pb2.Chunk(data=data)
            except Exception as e:
                print('Exception:', e)
            print('yield chunk (size: %d)' % len(data))
            yield chunk
            bytes_ = bytes_[GRPC_CHUNK_SIZE:]
            print('yield done (left: %d)' % len(bytes_))
        print('End of _iterate')


def main():
    learner = LearnerServer()
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    learner_pb2_grpc.add_LearnerServiceServicer_to_server(learner, server)
    server.add_insecure_port('[::]:61084')
    server.start()
    server.wait_for_termination()