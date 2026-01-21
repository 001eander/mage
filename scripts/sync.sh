#!/bin/bash

# Configuration
# Allow override via environment: REMOTE_SERVER, REMOTE_DIR
remote_server="${REMOTE_SERVER:-SUFE}"
remote_dir="${REMOTE_DIR:-~/mage}"
watch_mode="${1:-once}"  # 'once' or 'watch'

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN:${NC} $1"
}

# Sync function for local to remote
sync_to_remote() {
    log "Syncing local project to remote..."
    ssh "$remote_server" "mkdir -p $remote_dir" || { error "Failed to create remote directory"; return 1; }

    # Use rsync with optimized flags
    # -a: archive mode, -z: compress, -h: human-readable, --delete: remove deleted files
    # Exclude local envs, caches, datasets and outputs
    rsync -azh --delete \
        --exclude='.git/' \
        --exclude='.gitkeep' \
        --exclude='.venv/' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.pt' \
        --exclude='*.pth' \
        --exclude='*.ckpt' \
        --exclude='output_dir/' \
        --exclude='output/' \
        --exclude='data/' \
        --exclude='mage.egg-info/' \
        ./ "$remote_server:$remote_dir/" || { error "Project sync failed"; return 1; }

    log "Local to remote sync completed"
}

# Sync function for remote to local
sync_from_remote() {
    log "Syncing remote outputs back to local..."
    # MAGE uses output_dir for checkpoints and logs
    ssh "$remote_server" "mkdir -p $remote_dir/output_dir" || { error "Failed to ensure remote output_dir"; return 1; }

    rsync -azh --delete --exclude='.gitkeep' \
        "$remote_server:$remote_dir/output_dir/" ./output_dir/ || warn "Failed to sync output_dir/"

    log "Remote to local sync completed"
}

# Full sync
full_sync() {
    sync_to_remote
    sync_from_remote
}

# Watch mode with inotify
watch_sync() {
    # Check if inotifywait is installed
    if ! command -v inotifywait &> /dev/null; then
        error "inotifywait not found. Please install inotify-tools:"
        error "  Ubuntu/Debian: sudo apt-get install inotify-tools"
        error "  Fedora/RHEL: sudo dnf install inotify-tools"
        exit 1
    fi
    
    log "Starting watch mode (real-time sync)..."
    log "Watching directories: config/, scripts/, taming/, util/ and top-level files"
    log "Press Ctrl+C to stop"
    
    # Initial full sync
    full_sync
    
    # Background job for periodic remote to local sync
    (
        while true; do
            sleep 30  # Check every 30 seconds
            sync_from_remote
        done
    ) &
    remote_sync_pid=$!
    
    # Trap to cleanup on exit
    trap "kill $remote_sync_pid 2>/dev/null; log 'Watch mode stopped'; exit" SIGINT SIGTERM
    
    # Watch for local changes and sync to remote
    while true; do
        inotifywait -r -e modify,create,delete,move \
            --exclude '__pycache__|\.pyc$|\.git|\.venv' \
            config/ scripts/ taming/ util/ \
            engine_finetune.py engine_pretrain.py gen_img_uncond.py \
            main_finetune.py main_linprobe.py main_pretrain.py \
            models_mage.py models_vit_mage.py prepare_imgnet_val.py \
            setup.py pyproject.toml README.md environment.yaml 2>/dev/null

        if [ $? -eq 0 ]; then
            log "Changes detected, syncing to remote..."
            sleep 1  # Brief delay to batch rapid changes
            sync_to_remote
        fi
    done
}

# Main execution
case "$watch_mode" in
    watch)
        watch_sync
        ;;
    once|*)
        full_sync
        ;;
esac
