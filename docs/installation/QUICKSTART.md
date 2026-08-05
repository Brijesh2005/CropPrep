# Quickstart

Get the full CropFusion stack running with Docker in about five minutes.

## 1. Clone and configure

```bash
git clone <your-cropfusion-repo> cropfusion
cd cropfusion
cp .env.example .env
```

## 2. Start the stack

```bash
docker compose up -d
```

This builds and starts the frontend, backend, Postgres+PostGIS, Redis,
documentation site, Prometheus, Grafana and Loki.

## 3. Open the apps

| What | URL |
|---|---|
| CropFusion UI | http://localhost:3000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Documentation site | http://localhost:8080 |
| Grafana | http://localhost:3001 (admin / admin) |
| Prometheus | http://localhost:9090 |

## 4. Make your first prediction

```bash
curl -s http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"lon": 74.8, "lat": 13.1, "year": 2020, "season": "Kharif"}'
```

## 5. Optional extras

```bash
docker compose --profile mlops up -d     # MLOps scheduler (drift/fairness)
docker compose --profile devtools up -d  # pgAdmin on :5050
```

## 6. Stop

```bash
docker compose down            # keeps volumes
docker compose down -v         # removes volumes (data loss!)
```

## Next steps

- [DEPLOYMENT.md](DEPLOYMENT.md) — production deployment
- [API.md](API.md) — explore the API
- [manuals/FARMER_GUIDE.md](manuals/FARMER_GUIDE.md) — using the app
