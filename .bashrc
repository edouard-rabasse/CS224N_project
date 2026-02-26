# À ajouter dans ~/.bashrc ou ~/.zshrc
activate_env() {
    local VENV_PATH="/Data/edouard.rabasse/venvs/CS224N_project/bin/activate"
    
    if [ -f "$VENV_PATH" ]; then
        source "$VENV_PATH"
        echo "✅ Environnement CS224N_project activé."
        VENV_BASE_DIR="/Data/edouard.rabasse/venvs/"
        PROJECT_NAME=$(basename "$PWD")
        TARGET_VENV_PATH="$VENV_BASE_DIR/$PROJECT_NAME"
        export UV_PROJECT_ENVIRONMENT="$TARGET_VENV_PATH"
        echo "📍 UV_PROJECT_ENVIRONMENT set to: $TARGET_VENV_PATH"

    else
        echo "❌ Erreur : Le venv est introuvable à l'adresse $VENV_PATH"
        return 1
    fi
}