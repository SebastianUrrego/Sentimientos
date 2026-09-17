# Infraestructura AWS — MLflow Tracking Server + Postgres

Este documento registra cómo se levantó la infraestructura del laboratorio
(Análisis de Sentimientos, PLN) sobre un **AWS Academy Learner Lab**. Sirve
como runbook para el equipo y como evidencia de procedencia para la
evaluación.

## Recursos creados manualmente en la consola de AWS

| Recurso | Nombre / valor | Notas |
|---|---|---|
| Bucket S3 | `nlp-lab2-sentimientos` | Artifact store de MLflow. Versionado activado, acceso público bloqueado. |
| Security Group | `nlp-lab2-sentiment140-sg` | Entrada: 22 (solo IP propia), 5000 (MLflow, 0.0.0.0/0), 8000 (API, 0.0.0.0/0). Salida: All traffic 0.0.0.0/0. |
| Key pair | `nlp-lab2-sentiment140-key` | `.pem` — nunca se sube al repo. |
| Instancia EC2 | `nlp-lab2-sentiment140-mlflow-host` (t3.small, Ubuntu) | IAM instance profile: `LabInstanceProfile` (el que provee el Learner Lab, sin crear roles nuevos). |
| Elastic IP | asociada a la instancia | URL pública fija para MLflow (y luego la API). |

### Detalle importante del lanzamiento de la EC2

Al lanzar la instancia, en **Advanced details**:

- **IAM instance profile**: `LabInstanceProfile`.
- **Metadata version**: V2 only (token required).
- **Metadata token response hop limit**: **2** (necesario para que el
  contenedor Docker de MLflow pueda tomar credenciales del rol de la
  instancia vía IMDS y así hablarle a S3 sin claves hardcodeadas — con
  hop limit 1, el salto extra de red de Docker lo bloquea).

## Software dentro de la instancia

Ubuntu + Docker CE (instalado desde el repositorio oficial de Docker, no
el paquete `docker.io` de Ubuntu) + plugin `docker compose`.

Los archivos de este directorio (`Dockerfile.mlflow`, `docker-compose.yml`)
se copian a `/opt/mlflow-stack/` en la instancia. El backend store de
MLflow es Postgres (contenedor propio, no RDS — no siempre está habilitado
en Academy); el artifact store es el bucket S3 de arriba, vía el rol de la
instancia (no hay credenciales AWS hardcodeadas en ningún lado).

## Cómo desplegar / actualizar

En la EC2 (`ssh -i nlp-lab2-sentiment140-key.pem ubuntu@<ELASTIC_IP>`):

```bash
cd /opt/mlflow-stack
cp .env.example .env        # solo la primera vez
nano .env                   # pon el POSTGRES_PASSWORD real (openssl rand -hex 16) y el bucket
docker compose up -d --build
```

`restart: always` hace que los contenedores vuelvan a levantar solos si se
reinicia la instancia (por ejemplo, al pararla/arrancarla para ahorrar
créditos del Lab).

## Verificación rápida

```bash
docker compose ps
docker compose exec mlflow cat /proc/1/cmdline | tr '\0' '\n'   # confirma --host 0.0.0.0
docker compose exec mlflow python3 -c "import boto3; print(boto3.client('sts').get_caller_identity())"
```

Y desde cualquier navegador: `http://<ELASTIC_IP>:5000`.

## Experiment de MLflow

Todos los runs del equipo se registran bajo el Experiment de nombre exacto
`nlp-lab2-sentiment140` (creado manualmente desde la UI de MLflow), tal
como exige la Sección 6 de la guía del laboratorio.

## Pendiente

- Agregar el servicio `api:` (comentado en `docker-compose.yml`) cuando
  exista la imagen de FastAPI.
- Nginx + autenticación básica de solo lectura para el evaluador, una vez
  esté la API.
