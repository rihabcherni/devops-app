### Mini Projet: Architecture Docker pour la Classification des Genres Musicaux

#### Université de Tunis, École nationale supérieure d'ingénieurs de Tunis (ENSIT)
- **Projet**: Nouvelles Architectures
- **Date de remise**: 12 Décembre 2024
- **Section**: 3GInfo

---

### Structure des fichiers et des dossiers

Voici une proposition de structure de fichiers et de dossiers pour ce projet, qui inclut les services Flask, le frontend, Docker Compose, et l'intégration avec Jenkins.

```plaintext
project-root/
│
├── data/                       # Dataset et fichiers WAV pour les tests
│   └── gtzan/                  # Dataset GTZAN téléchargé (organisé par genre)
│
├── services/
│   ├── SVM_service/            # Service Flask utilisant le modèle SVM
│   │   ├── app.py              # Application Flask pour le service SVM
│   │   ├── model/              # Modèles ML (ex. SVM) et prétraitement des données
│   │   │   └── svm_model.pkl   # Modèle SVM pré-entrainé
│   │   ├── Dockerfile          # Dockerfile pour le service SVM
│   │   └── requirements.txt    # Dépendances Python pour le service SVM
│   │
│   ├── VGG19_service/          # Service Flask utilisant le modèle VGG19
│   │   ├── app.py              # Application Flask pour le service VGG19
│   │   ├── model/              # Modèles ML (ex. VGG19) et prétraitement des données
│   │   │   └── vgg19_model.h5  # Modèle VGG19 pré-entrainé
│   │   ├── Dockerfile          # Dockerfile pour le service VGG19
│   │   └── requirements.txt    # Dépendances Python pour le service VGG19
│
├── frontend/                   # Application frontend (ex. React, Vue, etc.)
│   ├── src/
│   │   ├── components/         # Composants front-end pour l'interface utilisateur
│   │   ├── App.js              # Fichier principal de l'application
│   │   ├── index.js            # Point d'entrée de l'application
│   ├── public/
│   │   └── index.html          # Page HTML principale
│   ├── Dockerfile              # Dockerfile pour le frontend
│   └── package.json            # Dépendances du frontend
│
├── orchestrator/               # Conteneur de coordination des services Flask
│   ├── app.py                  # Application pour appeler les deux services ML
│   ├── Dockerfile              # Dockerfile pour le conteneur d'orchestration
│   └── requirements.txt        # Dépendances Python pour le conteneur d'orchestration
│
├── jenkins/
│   ├── Dockerfile              # Dockerfile pour configurer Jenkins
│   └── jenkinsfile             # Script Jenkins pour CI/CD
│
├── docker-compose.yml          # Fichier de configuration pour orchestrer tous les conteneurs
├── README.md                   # Instructions pour installer et lancer le projet
└── report/                     # Rapport final du projet
    └── rapport.pdf             # Rapport PDF de 10 pages
```

---

### Détails des Contenus

1. **data**:
   - Stocke le dataset GTZAN et d'autres fichiers nécessaires pour les tests.

2. **services/SVM_service**:
   - Contient le service Flask pour la classification des genres musicaux utilisant un modèle SVM.
   - `app.py`: endpoint Flask pour prédire le genre musical à partir d'un fichier `wav` en base64.
   - `model/svm_model.pkl`: modèle SVM pré-entrainé pour la classification.
   - `Dockerfile`: installe les dépendances, expose le port du service.

3. **services/VGG19_service**:
   - Contient le service Flask pour la classification des genres musicaux utilisant un modèle VGG19.
   - `app.py`: endpoint Flask pour prédire le genre musical à partir d'un fichier `wav` en base64.
   - `model/vgg19_model.h5`: modèle VGG19 pré-entrainé.
   - `Dockerfile`: installe les dépendances, expose le port du service.

4. **frontend**:
   - Frontend simple pour appeler les services Flask et afficher les résultats de la classification.
   - `Dockerfile`: construit et sert l'application.

5. **orchestrator**:
   - Conteneur qui orchestre les appels aux services SVM et VGG19.
   - `app.py`: contient les routes pour appeler et combiner les résultats des deux services.

6. **jenkins**:
   - Contient le Dockerfile et les configurations pour Jenkins.
   - `jenkinsfile`: script Jenkins pour automatiser l'intégration, le déploiement et les tests des services.

7. **docker-compose.yml**:
   - Décrit les services, volumes et réseaux pour orchestrer les conteneurs des services ML, frontend, orchestrateur et Jenkins.

---

### Instructions de Déploiement

1. **Build et lancement des conteneurs**:
   ```bash
   docker-compose up --build
   ```
   Ce fichier compose orchestre les services SVM, VGG19, frontend, orchestrateur et Jenkins.

2. **Accès aux services**:
   - Frontend: `http://localhost:3000`
   - SVM_service: `http://localhost:<port_svm>`
   - VGG19_service: `http://localhost:<port_vgg19>`
   - Jenkins: `http://localhost:8080`

3. **Jenkins CI/CD**:
   - Jenkins déploie les conteneurs automatiquement, teste les endpoints, et peut déployer des versions à jour.

4. **Tests de l'application**:
   - Utiliser Jenkins pour exécuter des tests automatisés et valider le fonctionnement de chaque service.

---

### Remarque sur le Rapport

Le rapport de 10 pages détaillera:
- L'architecture Docker mise en place
- Les choix de modèles et méthodes ML
- Les services web Flask et leur implémentation
- Les étapes pour configurer et déployer l'application
- Les résultats des tests et des évaluations# devops-app
