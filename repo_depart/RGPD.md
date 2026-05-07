# Conformité RGPD — ChurnGuard

## Contexte

ChurnGuard est un système de prédiction de résiliation (churn) client développé pour TelcoFr. Il traite des données à caractère personnel de clients d'un opérateur télécom dans le cadre d'une mission d'industrialisation MLOps.

---

## 1. Données traitées

Le modèle utilise le dataset **Telco Customer Churn** (IBM Sample Data, usage éducatif).

| Catégorie | Champs |
|---|---|
| Profil client | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Ancienneté | `tenure` |
| Services souscrits | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Contrat & facturation | `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Label (entraînement) | `Churn` (Yes/No) |

Ces données sont des **données à caractère personnel** au sens de l'article 4 du RGPD car elles permettent, combinées, d'identifier indirectement un individu.

---

## 2. Finalité du traitement

| Finalité | Base légale (Art. 6 RGPD) |
|---|---|
| Entraînement du modèle prédictif | Intérêt légitime de l'entreprise (Art. 6.1.f) |
| Prédiction du risque de résiliation | Exécution du contrat / intérêt légitime (Art. 6.1.b/f) |
| Amélioration de la rétention client | Intérêt légitime (Art. 6.1.f) |

---

## 3. Décision automatisée — Article 22 RGPD

Le modèle produit une **prédiction probabiliste** (`churn: true/false`, `probability: 0.0–1.0`). Ce score est destiné à **assister** un conseiller humain, et non à constituer une décision automatisée définitive ayant des effets juridiques sur le client.

> Aucune décision contractuelle (résiliation, modification d'offre, sanction) ne doit être prise sur la seule base du score produit par l'API.

Si le système évolue vers une décision automatisée au sens de l'Art. 22, une **analyse d'impact relative à la protection des données (AIPD)** devra être conduite préalablement.

---

## 4. Durée de conservation

| Donnée | Durée |
|---|---|
| Dataset d'entraînement | Durée du projet — suppression à l'issue de l'ECF |
| Modèle entraîné (MLflow) | 3 ans maximum (durée de pertinence du modèle) |
| Logs d'API (prédictions) | Non stockés par défaut — aucune donnée client persistée |
| Métriques MLflow (runs) | Sans données personnelles — conservation illimitée |

---

## 5. Droits des personnes concernées

Conformément aux articles 15 à 22 du RGPD, tout client dont les données sont utilisées dispose des droits suivants :

- **Droit d'accès** (Art. 15) : connaître les données le concernant traitées par le modèle.
- **Droit de rectification** (Art. 16) : corriger des données inexactes.
- **Droit à l'effacement** (Art. 17) : demander la suppression des données d'entraînement.
- **Droit d'opposition** (Art. 21) : s'opposer au profilage.
- **Droit à l'explication** (Art. 22) : obtenir une explication sur la logique du modèle.

Point de contact DPO : à désigner par TelcoFr avant mise en production.

---

## 6. Mesures techniques et organisationnelles

| Mesure | Implémentation |
|---|---|
| Pseudonymisation | `customerID` supprimé avant entraînement (`data.preprocess`) |
| Pas de stockage des prédictions | L'API ne persiste aucune donnée client |
| Accès restreint | Conteneur API en utilisateur non-root |
| Traçabilité des modèles | Versioning MLflow avec métriques et paramètres |
| Pas de transfert hors UE | Dataset et modèle stockés localement |
| Sécurité des dépendances | Scan Trivy à chaque build CI |

---

## 7. Données utilisées pour ce projet

Le dataset utilisé est **public** (IBM Sample Data, Telco Customer Churn), mis à disposition à des fins éducatives et de recherche. Il est **pseudonymisé** : aucun nom, adresse ou numéro de téléphone réel n'est présent.

En contexte de production réel chez TelcoFr, le traitement de données clients réelles nécessiterait :

1. Une base légale explicite (consentement ou intérêt légitime documenté).
2. Une information préalable des clients (mention dans les CGU / politique de confidentialité).
3. La désignation d'un DPO si le traitement est à grande échelle.
4. Une AIPD si le traitement présente un risque élevé (Art. 35 RGPD).

---

*Document établi dans le cadre de l'ECF MLOps — RNCP 35288 bloc 5.*
