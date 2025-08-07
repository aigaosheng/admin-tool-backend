## Install

### Install backend

- Set .env

```
MODEL_NAME = "gemini-1.5-flash"
LLM_PROVIDER = "gemini"
GOOGLE_APPLICATION_CREDENTIALS="gcloud-gemini-key.json"
HOST=0.0.0.0
PORT=8000
```

- Backend deploy docker

```
# Local run

uvicorn main:app --reload --host 0.0.0.0 --port 8000

```


```
# Build docker

docker build -t taxinvoice:latest .

docker run --rm -it -p 8000:8000 taxinvoice
docker run --rm -it -p 8000:8000 taxinvoice --network="host" 

```

```
# push to Docker hub

docker tag taxinvoice shenggs/taxinvoice:latest

docker push shenggs/taxinvoice

```

- Frontend deploy

```
# local run

flutter run -d web-server --web-port 3000

```


```
flutter create . --platforms web #web app

flutter clean

flutter pub get

flutter build web --release

ls build/web/index.html #OK

```

```

#deploy in vercel
npm install -g vercel
vercel --prod build/web

```
