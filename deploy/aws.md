# Deploying to AWS (ECR + ECS Fargate)

Replace `ACCOUNT`, `REGION` and names as needed.

## 1. Push the image to ECR
```bash
export REGION=eu-central-1 ACCOUNT=123456789012 REPO=rag-orchestrator
aws ecr create-repository --repository-name $REPO --region $REGION
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker build --platform linux/amd64 -t $REPO .
docker tag $REPO:latest $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
```

## 2. Store secrets in Secrets Manager
```bash
aws secretsmanager create-secret --name rag/groq     --secret-string "gsk_..." --region $REGION
aws secretsmanager create-secret --name rag/pinecone --secret-string "pcsk_..." --region $REGION
```

## 3. ECS task definition (Fargate), key parts
```json
{
  "family": "rag-orchestrator",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512", "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [{
    "name": "api",
    "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/rag-orchestrator:latest",
    "portMappings": [{ "containerPort": 8000 }],
    "environment": [
      { "name": "PINECONE_INDEX", "value": "rag-orchestrator-free" },
      { "name": "LLM_MODEL", "value": "openai/gpt-oss-120b" }
    ],
    "secrets": [
      { "name": "GROQ_API_KEY",     "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:rag/groq" },
      { "name": "PINECONE_API_KEY", "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:rag/pinecone" }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": { "awslogs-group": "/ecs/rag-orchestrator", "awslogs-region": "REGION", "awslogs-stream-prefix": "api" }
    }
  }]
}
```
The execution role needs `secretsmanager:GetSecretValue` on both secrets.

## 4. Service behind an Application Load Balancer
- Create a cluster, then an ECS service (Fargate) from the task definition, 2 tasks.
- ALB target group: protocol HTTP, port 8000, health check path `/health`.
- Security groups: ALB allows 443/80 from the internet; tasks allow 8000 only from the ALB.
- Add an HTTPS listener with an ACM certificate.

## 5. Redeploy
```bash
aws ecs update-service --cluster rag --service rag-orchestrator --force-new-deployment
```