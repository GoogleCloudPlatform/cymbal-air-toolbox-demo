# Running the LangChain Demo

##  Before you begin

1. Make sure you've [setup and initialized your
   Database](../README.md#setting-up-your-database).

1. Make sure you've [deployed your extension service, and are running a
   connection to it locally on
   127.0.0.1:8080](../README.md#deploying-the-extension-service).

1. Make sure you have Python 3.11+ installed

## Setting up your Environment

1. Set up [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials#GAC):

```bash
    gcloud auth application-default login
```

1. Change into the `langchain_tools_demo` directory:

```bash
    cd langchain_tools_demo
```

1. Install the dependencies using `pip`. You may wish to do this in a
   [venv](https://docs.python.org/3/library/venv.html):

```bash
    pip install -r requirements.txt
```


## Running the Demo

1. Start the application with:

```
    python main.py
```
Note: for hot reloading of the app use: `uvicorn main:app --host 0.0.0.0 --reload`

1. View app at `http://localhost:8081/`
