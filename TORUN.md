# TORUN.md — Pipeline complet : entraînement → évaluation → comparaison

Commandes à lancer dans l'ordre sur le cluster GPU (Sherlock / Azure).

---

## 0. Prérequis

```bash
# Activer l'environnement
source activate_env.sh   # ou équivalent sur le cluster

# Vérifier que les données sont là
ls data/wiki1m_for_simcse.txt        # 1M phrases Wikipedia
ls data/parsed_graphs/               # graphes de dépendance pré-parsés

# Si les graphes ne sont pas encore parsés (long — plusieurs heures)
uv run python scripts/parse_graphs.py \
    --input data/wiki1m_for_simcse.txt \
    --output data/parsed_graphs/
```

---

## 1. Entraînement

Trois stratégies disponibles. Lancer les 3 pour comparer (ou choisir `multi_loss` en priorité).

```bash
# Stratégie 1 : joint training dès l'époch 0 (recommandée)
uv run python src/train.py experiment=multi_loss

# Stratégie 2 : stop-gradient (GNN figé, aligne BERT uniquement)
uv run python src/train.py experiment=stop_grad

# Stratégie 3 : freeze-then-align (warmup GNN puis joint)
uv run python src/train.py experiment=freeze_then_align
```

Les checkpoints sont sauvegardés dans :
- `outputs/multi_loss/`
- `outputs/stop_grad/`
- `outputs/freeze_then_align/`

---

## 2. Évaluation de notre modèle

```bash
# Évaluer le meilleur checkpoint (ex: multi_loss) sur toutes les tâches STS
uv run python src/evaluate.py \
    --model-path outputs/multi_loss \
    --mode test \
    --task-set sts \
    --output-json outputs/our_model_sts.json

# Optionnel : tâches de transfert (classification)
uv run python src/evaluate.py \
    --model-path outputs/multi_loss \
    --mode test \
    --task-set transfer \
    --output-json outputs/our_model_transfer.json
```

---

## 3. Comparaison avec les baselines

```bash
# BERT-base + SimCSE-BERT sur toutes les tâches STS (+ notre modèle si dispo)
uv run python scripts/run_baselines.py \
    --mode test \
    --task-set sts \
    --our-model outputs/multi_loss \
    --output-json outputs/comparison_sts.json

# Avec les tâches de transfert aussi
uv run python scripts/run_baselines.py \
    --mode test \
    --task-set full \
    --our-model outputs/multi_loss \
    --output-json outputs/comparison_full.json
```

Résultats attendus (référence papier SimCSE, mode `test`) :

| Task         | BERT-base | SimCSE-BERT | Ours |
|--------------|-----------|-------------|------|
| STS12        | ~55       | 68.40       | ?    |
| STS13        | ~59       | 82.41       | ?    |
| STS14        | ~57       | 74.38       | ?    |
| STS15        | ~61       | 80.91       | ?    |
| STS16        | ~58       | 78.56       | ?    |
| STSBenchmark | ~59       | 76.85       | ?    |
| SICKRelat.   | ~64       | 72.23       | ?    |
| **Avg.**     | **~59**   | **76.25**   | ?    |

> Note : les scores BERT-base ci-dessus utilisent `avg_first_last` (pooler recommandé par SimCSE pour vanilla BERT).
> Les scores SimCSE-BERT utilisent `cls_before_pooler` (pooler à l'inférence, MLP head écarté).
> Source : [SimCSE paper Table 1](https://arxiv.org/pdf/2104.08821.pdf) + [SimCSE README](SimCSE/README.md).

---

## 4. Résultats déjà calculés

Les baselines ont déjà été partiellement calculées en local (CPU, mode `fasttest`, STSBenchmark uniquement) :

```
outputs/baseline_comparison.json   ← BERT-base: 20.29 (cls_before_pooler, à recalculer)
                                       SimCSE-BERT: 76.52 ✓
```

> **À refaire sur GPU avec `--mode test --task-set sts`** pour avoir les vrais scores sur toutes les tâches.
> Le score BERT-base de 20.29 est avec `cls_before_pooler` (mauvais pooler) — il sera ~59 avec `avg_first_last`.
