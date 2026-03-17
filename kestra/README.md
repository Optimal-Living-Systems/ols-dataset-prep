# Kestra Orchestration

This directory contains the Phase 4 production Kestra flow for `ols-dataset-prep`.

## Import into Kestra UI

1. Open Kestra at `http://localhost:8080`.
2. Go to `Flows`.
3. Click `Create`, then switch to the code editor.
4. Paste the contents of `kestra/ols-dataset-prep-flow.yml`.
5. Save the flow in namespace `ols.data`.

You can also deploy from the CLI:

```bash
kestra flow update ./kestra/ols-dataset-prep-flow.yml ols.data ols-dataset-prep --server http://localhost:8080
```

If your Kestra instance has authentication enabled, add `--user <username:password>`.

## Set Kestra Secrets

Create these secrets before you run the flow:

- `HF_TOKEN`
- `ANTHROPIC_API_KEY`
- `OLS_DATASET_PREP_WEBHOOK_KEY`

In the UI:

1. Open `Namespaces`.
2. Select `ols.data`.
3. Open the `Secrets` tab.
4. Add each secret key/value pair.

CLI examples:

```bash
kestra namespace secret create ols.data HF_TOKEN "<your-hf-token>" --server http://localhost:8080
kestra namespace secret create ols.data ANTHROPIC_API_KEY "<your-anthropic-key>" --server http://localhost:8080
kestra namespace secret create ols.data OLS_DATASET_PREP_WEBHOOK_KEY "<random-webhook-key>" --server http://localhost:8080
```

## Trigger Manually

From the UI:

1. Open `ols.data.ols-dataset-prep`.
2. Click `Execute`.
3. Optionally set `dataset_id`, `force`, or `augment`.
4. Start the execution.

From a webhook:

```bash
curl -X POST \
  http://localhost:8080/api/v1/main/executions/webhook/ols.data/ols-dataset-prep/<OLS_DATASET_PREP_WEBHOOK_KEY>
```

If Basic Auth is enabled on your Kestra instance, send an `Authorization` header as well.

## Add a New Dataset

1. Edit `config/datasets.yaml` and add the new dataset entry.
2. Run the flow manually from Kestra, or wait for the Monday 6:00 AM schedule.
3. If you use the file trigger, drop a `.yaml` file into `/home/joel/ols-pipeline/new-datasets/` to start a new execution.

If you set `augment` to `false`, the flow calls `ols-prep run ... --no-augment` and skips Stage 3.

## Docker Note

This flow uses the Process task runner and the local filesystem trigger. Because Kestra is running in Docker, the repo path, output path, and `/home/joel/ols-pipeline/new-datasets/` must be bind-mounted into the Kestra container so those absolute paths resolve inside the container as well.
