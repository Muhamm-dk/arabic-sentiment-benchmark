# Chapter 18: Publishing and deployment

## What

| Artifact | Where | Contents |
|---|---|---|
| Model | Hugging Face Hub, `Muhamm-dk/arabic-sentiment-marbert-combined` | weights (651 MB), tokenizer, config, model card |
| Image | GitHub Container Registry, `ghcr.io/muhamm-dk/arabic-sentiment-api:0.2.0` | code and libraries, no weights (Chapter 17) |
| Service | Azure Container Apps, UAE North | the image, with weights fetched from the Hub at start |
| Code, guide | GitHub, `Muhamm-dk/arabic-sentiment-benchmark` | everything else |

The live service is at `https://arabic-sentiment-api.niceflower-4ddb1508.uaenorth.azurecontainerapps.io`. It is a portfolio demo: one replica at most, a shared key, no rate limiting, and a student credit behind it.

## Why this split

- **Weights on the Hub.** They are 651 MB and a registry image of that size is slow to build and push. A Hub repository is versioned, free for public models (from memory), and has a model card page that doubles as portfolio material.
- **Image on GitHub's registry, public.** Free for public images (from memory, not checked) and next to the code. It was made public so Azure could pull it with no registry credentials. The proof is in the Azure output: `"registries": null` and the app started. The image holds nothing secret.
- **Azure Container Apps, consumption plan, scale to zero.** It runs the existing image without a server to manage, and with no traffic it should use no compute. I picked it because the student credit covers Azure and because it has a monthly free grant.
- **API key in a platform secret,** not in the image or the repository.

## Alternatives

Nothing here was run except what is marked.

| Option | Why not |
|---|---|
| Hugging Face Docker Space | Not selectable on the free account (Docker and Gradio showed as paid). I had called it free from memory, and I was wrong. Static Spaces cannot run a Python service |
| Azure App Service, AKS, a small VM | Not tried. The free VM sizes in the free-services list are described as burstable and I believed their memory too small for this model (from memory, unverified) |
| Oracle Cloud always-free VM | Not tried. Needs a card and more manual setup |
| AWS | Not tried. Free-tier memory was my concern (unverified) |
| Render, Railway, Fly.io | Not checked |
| Weights baked into the image | Removes the 27 s download at cold start, adds about 650 MB to the image |
| Always-on replica | Removes cold starts and would use up the free grant in days |
| Azure Container Registry | Costs credit unless covered by a free grant. The public registry was enough |
| No live host | A fully acceptable end state, and the fallback if the credit ends |

## How

### 1. The model on the Hub

```python
api.create_repo("Muhamm-dk/arabic-sentiment-marbert-combined", exist_ok=True)
api.upload_folder(folder_path="outputs/combined-marbert-s0", repo_id=repo)
```

- A write token created in the account settings, entered through a hidden prompt (`read -rs`) and unset afterwards. It never entered the repository or the chat.
- `upload_folder` sent 5 files: `model.safetensors` (651 MB), `config.json`, `tokenizer.json`, `tokenizer_config.json` and `README.md`. There is no `vocab.txt`, because the fast tokenizer is fully in `tokenizer.json`.
- The first `pipeline(...)` call returned `LABEL_0` with score 0.9845. The config had only generic label names. The fix wrote `id2label` and `label2id` into `config.json` and re-uploaded only that file:

```python
c["id2label"] = {"0": "negative", "1": "positive"}
c["label2id"] = {"negative": 0, "positive": 1}
```

  After that the same sentence returned `negative` at 0.9845. The API maps labels itself (`LABELS` in `app.py`), so it was never affected, but the model page widget and direct users were.
- **The model card** is the first thing a reader sees. It was expanded after the first upload to add training size, procedure, intervals, the LABR cost against the single-dataset model, the overconfidence of the score, the HARD rule flaw and the untested domains. A copy lives in `docs/model_card.md`. The license field is `other`, because the datasets state none (Chapter 3).
- Not pinned: the container loads the latest revision of this repository. A re-upload changes what new containers load.

### 2. The image on GitHub's registry

```bash
echo $CR_PAT | docker login ghcr.io -u Muhamm-dk --password-stdin
docker tag arabic-sentiment-api ghcr.io/muhamm-dk/arabic-sentiment-api:0.2.0
docker push ghcr.io/muhamm-dk/arabic-sentiment-api:0.2.0
```

- A classic personal access token with only the `write:packages` scope, entered through a hidden prompt and unset afterwards.
- The package visibility was set to Public in GitHub's package settings. Check: `docker logout ghcr.io` then `docker pull` returned the digest `sha256:6f4c049769d6f17aca7bd1a4f73a422936601725bdf9a6fa1b5fbc7415052a66`, and it said the image was up to date because the layers were already local. The registry call still happened without credentials, which a private package would have refused.
- The tag `0.2.0` is mutable. Pushing the same tag again would change the image a new revision gets. A digest reference would not.
- `0.2.0` was the first push with the key check in the code (an earlier build had none), and it is the only version deployed.

### 3. The service on Azure

Set-up, in order, from the terminal (Azure CLI 2.91.0 installed with Microsoft's script):

```bash
az login --use-device-code
az account show --query "{name:name, state:state}" -o table      # Azure for Students, Enabled
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az group create -n rg-arabic-sentiment -l uaenorth
az containerapp env create -n env-arabic-sentiment -g rg-arabic-sentiment -l uaenorth
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > ~/.sentiment_api_key
chmod 600 ~/.sentiment_api_key
az containerapp create -n arabic-sentiment-api -g rg-arabic-sentiment \
  --environment env-arabic-sentiment \
  --image ghcr.io/muhamm-dk/arabic-sentiment-api:0.2.0 \
  --target-port 8000 --ingress external \
  --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1 \
  --secrets api-key=$(cat ~/.sentiment_api_key) \
  --env-vars API_KEY=secretref:api-key
```

What each choice does:

- **Providers.** `Microsoft.App` and `Microsoft.OperationalInsights` must be registered on a subscription once. Registration took a few minutes.
- **Region.** I tried `uaenorth` first because it is nearest. It worked, so the student subscription allows it.
- **Environment.** A Container Apps environment is the shared boundary for networking and logs. Creating it also created a Log Analytics workspace automatically (named `workspace-rgarabicsentimentLZLw`). That workspace may cost money (see below).
- **Sizing.** 1 vCPU and 2 GiB. The measured baseline was 405 MiB, so 2 GiB leaves margin I did not measure.
- **Scale.** Minimum 0 replicas (scale to zero), maximum 1. The default HTTP scale rule `http-scaler` was added automatically.
- **The key.** A random 43-character string stored in a file outside the repository (mode 600), passed once to Azure as a secret, and exposed to the container as the `API_KEY` environment variable through `secretref`. The CLI output shows only the secret's name, not its value. While the command runs, the value is visible in the process list on the laptop.
- **Ingress.** External, port 8000, HTTPS only (`allowInsecure: false`) with the platform's certificate. `corsPolicy` and `ipSecurityRestrictions` are null: anyone on the internet can reach it.

### 4. Tests against the live URL

```bash
curl -s https://$URL/health
curl -s -o /dev/null -w "no key: %{http_code}\n" -X POST https://$URL/predict -H "Content-Type: application/json" -d '{"text": "ممتاز"}'
curl -s -X POST https://$URL/predict -H "Content-Type: application/json" -H "X-API-Key: $(cat ~/.sentiment_api_key)" -d '{"text": "الخدمة كانت سيئة جدا"}'
```

Results: `/health` returned `device: cpu`, the keyless call returned 401, and the keyed call returned `negative` at 0.9845, the same score as locally.

### Operating it

| Task | Command (not all run) |
|---|---|
| Logs | `az containerapp logs show -n arabic-sentiment-api -g rg-arabic-sentiment --tail 50` |
| Replica count | `az containerapp replica list -n arabic-sentiment-api -g rg-arabic-sentiment --query "length(@)"` (run: it printed 0 before the cold-start test) |
| Deploy a new image | push a new tag, then `az containerapp update -n arabic-sentiment-api -g rg-arabic-sentiment --image ghcr.io/muhamm-dk/arabic-sentiment-api:<tag>` (not run) |
| Remove everything | `az group delete -n rg-arabic-sentiment --yes --no-wait` (not run; I believe it also removes the Log Analytics workspace, which was created in the same group, unverified) |

## Numbers

| Item | Value | Notes |
|---|---|---|
| Cold start | 27.5 s and 26.0 s for `/health` | Two measurements, each after the replica count read 0 (Oct 30 and Nov 7). A third at another time of day is open |
| Warm request | 0.88 s for `/predict`, 0.75 s for `/health` | Round trip from the laptop, including HTTPS handshake and network |
| Local baseline memory | 405 MiB | `docker stats` after one request. The largest request was not measured |
| Azure size | 1 vCPU, 2.0 GiB | |
| Free grant (from the pasted free-services page; may differ for a student subscription) | 180,000 vCPU-seconds, 360,000 GiB-seconds, 2M requests per month | At 1 vCPU and 2 GiB both run out after 180,000 seconds of active time, about 50 hours |
| Estimated visits inside the grant | about 440 to 530 isolated visits a month | 180,000 divided by the 342 to 408 s a replica lived per isolated visit (measured below). Assumes all of that time is billed at the active rate, which I did not verify |
| Student credit | $100 for 12 months, no card (as the offer says) | Valid until about Oct 2027 |

Azure end-to-end check (Nov 6, `container_check.py`): 240 saved test rows through `/predict_batch`, 0 label disagreements, mean probability difference 0.00033, maximum 0.0067, identical to the local container run. 45.6 s in total from the laptop, about 5 rows per second, with the replica already warm. Scale-down (Nov 7): after one request the replica count read 1 for the first six polls and 0 from the seventh. The polls were about 65 s apart and the first came a few seconds after the request, so the replica was gone between about 5.7 and 6.8 minutes after it. That is consistent with a default cooldown of 5 minutes (from memory) plus polling delay. Second cold start, after the count read 0 for about 9 minutes: 26.0 s (the first was 27.5 s). Still to be filled: a third cold start at another time of day, cost to date, credit left. (fill in)

## Security review

| Aspect | State |
|---|---|
| Transport | HTTPS with the platform certificate |
| Authentication | One shared key. Fails open if the variable is unset (the Azure app sets it) |
| Key storage | Platform secret. A copy sits in a mode-600 file on the laptop |
| Rate limiting | None. An attacker with the key, or without it on `/docs`, `/openapi.json` and `/health`, costs compute. Wrong-key requests are cheap, valid ones load the model |
| Network exposure | Public, no IP restrictions |
| Image | Public, no secrets. Runs as a non-root user. Not scanned for vulnerabilities |
| Dependencies | Unpinned versions (Chapter 17) |
| Model supply chain | Weights come from a Hub repository under the same account, unpinned revision, no signature check |
| Cost control | A budget alert in Cost Management (set by the owner). Alerts do not stop spending, though the student subscription is disabled when the credit runs out |
| Logging and monitoring | Console logs only. No alerts on errors or latency |
| Data | Request texts are processed and not stored by the code. Platform request logs were not reviewed |

## What went wrong

| Item | Detail |
|---|---|
| Docker Spaces unavailable | I told the owner HF Docker Spaces were free from memory. The account could not select them. A claim from memory about a vendor's plans was wrong |
| `LABEL_0` on the Hub | See above |
| Student eligibility | The Azure for Students terms require full-time student status and the credit is single-use. The deployment depends on holding that eligibility. I do not have access to verify it |
| Cold start is a single sample | One measurement was used in the README until the checks are run |
| Possible hidden cost | The auto-created Log Analytics workspace. Not checked |
| Key typed in a chat once (Groq, not Azure) | Rotated and the repository scanned. Appendix A |

## Not done

- Pinning the image by digest and the model by revision, and a fail-closed key check.
- A rate limit, request logging review, an alert on errors or cost.
- Load testing, and measuring memory under the largest batch (Chapter 17 has the check).
- A smoke test that runs automatically after each deployment, and any CI pipeline.
- A decision about what happens when the credit ends or after 90 days of inactivity (the terms allow Microsoft to remove idle compute after notice and ask for a portal login or API access at least every 90 days, so log in before the end of January 2027).
- Baking the weights into the image, to avoid the cold-start download.

## Check yourself

1. Why is the model on the Hub and not inside the image, and what two risks does that create?
2. How do you know Azure pulled the image without credentials?
3. What does `--min-replicas 0` do to cost, and what does it do to the first request after idle?
4. Where is the API key stored, who can read it, and where could it leak?
5. What does "fails open" mean here, and what would "fails closed" look like?
6. Why is one cold-start measurement not enough, and what would you measure besides it?
7. What happens if you push a new image under the tag `0.2.0`?
8. What would you do when the student credit runs out?
