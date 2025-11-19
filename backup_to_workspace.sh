#!/bin/bash
# Backup ForeWatt models and MLflow to persistent /workspace storage
# Run this periodically or after training to preserve results

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/workspace/forewatt_backup"
SOURCE_DIR="/root/ForeWatt"

echo "==================================="
echo "ForeWatt Backup to Workspace"
echo "==================================="
echo "Timestamp: $TIMESTAMP"
echo "Source: $SOURCE_DIR"
echo "Destination: $BACKUP_DIR"
echo ""

# Create backup directories
mkdir -p "$BACKUP_DIR"/{mlflow,models,reports,checkpoints}

# Backup MLflow database and artifacts
echo "[1/5] Backing up MLflow..."
rsync -av --progress "$SOURCE_DIR/mlflow/" "$BACKUP_DIR/mlflow/" 2>/dev/null || echo "  No MLflow data yet"

# Backup trained models
echo "[2/5] Backing up models..."
rsync -av --progress "$SOURCE_DIR/models/" "$BACKUP_DIR/models/" 2>/dev/null || echo "  No models yet"

# Backup reports and results
echo "[3/5] Backing up reports..."
rsync -av --progress "$SOURCE_DIR/reports/" "$BACKUP_DIR/reports/" 2>/dev/null || echo "  No reports yet"

# Backup data (optional - comment out if too large)
# echo "[4/5] Backing up data..."
# rsync -av --progress "$SOURCE_DIR/data/gold/" "$BACKUP_DIR/data/gold/" 2>/dev/null || echo "  No gold data yet"

# Create backup manifest
echo "[5/5] Creating backup manifest..."
cat > "$BACKUP_DIR/backup_manifest_$TIMESTAMP.txt" << EOF
ForeWatt Backup Manifest
========================
Timestamp: $TIMESTAMP
Source: $SOURCE_DIR
Destination: $BACKUP_DIR

Contents:
$(du -sh $BACKUP_DIR/* 2>/dev/null)

MLflow runs:
$(ls -1 $BACKUP_DIR/mlflow/*.db 2>/dev/null | wc -l) database file(s)

Models:
$(find $BACKUP_DIR/models -name "*.pkl" 2>/dev/null | wc -l) model file(s)

Reports:
$(find $BACKUP_DIR/reports -name "*.csv" 2>/dev/null | wc -l) CSV file(s)
$(find $BACKUP_DIR/reports -name "*.json" 2>/dev/null | wc -l) JSON file(s)
EOF

echo ""
echo "✓ Backup complete!"
echo "Total backup size: $(du -sh $BACKUP_DIR | cut -f1)"
echo ""
echo "To restore:"
echo "  rsync -av $BACKUP_DIR/ $SOURCE_DIR/"
echo ""
