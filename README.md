![Logo]()

# Document AI - Backend

A brief description of what this project does and who it's for

## Deployment

To deploy this project run

## 🧰 Prerequisites

Make sure the following tools are installed:

- `git`
- `curl` or `wget`
- `build-essential` (on Linux)
- `pyenv` ([Install guide](https://github.com/pyenv/pyenv#installation))

```bash
# Install Python 3.11.8
pyenv install 3.11.8

# Set the local version for this project
pyenv local 3.11.8
```
## Install Dependencies

```bash
pip install -r requirements.txt
```

## Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Run LLama3.2

```bash
ollama run llama3.2
```


## Run Server

```bash
uvicorn main:app
```
or 
```bash
python main.py
```


## API Reference

You can access the Postman collection using the link below to test the API:

🔗 [Join the Postman team and access the collection](https://app.getpostman.com/join-team?invite_code=83bf12b44f4b6d0adb1189df65dbe985208f0893bca8508e0cd727c2d12e368b&target_code=20fa2128a1437503c0c30b3c3d634f36)


#### Upload Files

`http://127.0.0.1:8000`

```http
  POST /upload
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `file` | `file` | **Required** Uploads the File |

#### Question Prompt

```http
  GET /ask
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `session_id`      | `string` | **Required**. session_id generated after the upload |
| `question`|`string`|*Required* `question prompt asked by user with context`|


