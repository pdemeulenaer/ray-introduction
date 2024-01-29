# serve_fastapi_example.py

from fastapi import FastAPI
from ray import serve

app = FastAPI()

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class MyFastAPIDeployment:
    @app.get("/")
    def root(self):
        return "Hello, world!"

    @app.post("/{subpath}")
    def root(self, subpath: str):
        return f"Hello from {subpath}!"

my_fastapi_deployment = MyFastAPIDeployment.bind()
serve.run(my_fastapi_deployment, route_prefix="/hello")