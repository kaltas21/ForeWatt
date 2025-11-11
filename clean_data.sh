#!/bin/bash

# ForeWatt Data Cleanup Script
# Deletes data from medallion architecture layers (Bronze, Silver, Gold)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Data directory
DATA_DIR="./data"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  ForeWatt Data Cleanup Utility${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory '$DATA_DIR' not found${NC}"
    exit 1
fi

# Function to show directory size
show_size() {
    local dir=$1
    if [ -d "$dir" ]; then
        local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo -e "${YELLOW}  Size: $size${NC}"
    else
        echo -e "${YELLOW}  (Directory does not exist)${NC}"
    fi
}

# Function to delete directory
delete_dir() {
    local dir=$1
    local name=$2
    if [ -d "$dir" ]; then
        echo -e "${YELLOW}Deleting $name...${NC}"
        rm -rf "$dir"
        echo -e "${GREEN}✓ Deleted $name${NC}"
    else
        echo -e "${YELLOW}⚠ $name does not exist, skipping${NC}"
    fi
}

# Show current data structure and sizes
echo -e "${BLUE}Current data structure:${NC}"
echo ""

echo -e "${GREEN}Bronze Layer (Raw Data):${NC}"
echo "  1. Weather data: $DATA_DIR/bronze/demand_weather/"
show_size "$DATA_DIR/bronze/demand_weather"
echo "  2. EPİAŞ data: $DATA_DIR/bronze/epias/"
show_size "$DATA_DIR/bronze/epias"
echo ""

echo -e "${GREEN}Silver Layer (Normalized Data):${NC}"
echo "  3. Weather data: $DATA_DIR/silver/demand_weather/"
show_size "$DATA_DIR/silver/demand_weather"
echo "  4. EPİAŞ data: $DATA_DIR/silver/epias/"
show_size "$DATA_DIR/silver/epias"
echo ""

echo -e "${GREEN}Gold Layer (Feature-Engineered Data):${NC}"
echo "  5. Demand features: $DATA_DIR/gold/demand_features/"
show_size "$DATA_DIR/gold/demand_features"
echo ""

if [ -d "$DATA_DIR" ]; then
    echo -e "${BLUE}Total data directory size:${NC}"
    show_size "$DATA_DIR"
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${YELLOW}What would you like to delete?${NC}"
echo ""
echo "  1) Bronze layer only (all raw data)"
echo "  2) Silver layer only (all normalized data)"
echo "  3) Gold layer only (all features)"
echo "  4) Bronze + Silver layers"
echo "  5) All layers (bronze + silver + gold)"
echo "  6) Weather data only (all layers)"
echo "  7) EPİAŞ data only (all layers)"
echo "  8) Custom selection"
echo "  9) Cancel"
echo ""
read -p "Enter your choice (1-9): " choice

case $choice in
    1)
        echo ""
        echo -e "${RED}This will delete ALL bronze layer data${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/bronze" "Bronze layer"
        else
            echo "Cancelled."
        fi
        ;;
    2)
        echo ""
        echo -e "${RED}This will delete ALL silver layer data${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/silver" "Silver layer"
        else
            echo "Cancelled."
        fi
        ;;
    3)
        echo ""
        echo -e "${RED}This will delete ALL gold layer data${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/gold" "Gold layer"
        else
            echo "Cancelled."
        fi
        ;;
    4)
        echo ""
        echo -e "${RED}This will delete Bronze + Silver layers${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/bronze" "Bronze layer"
            delete_dir "$DATA_DIR/silver" "Silver layer"
        else
            echo "Cancelled."
        fi
        ;;
    5)
        echo ""
        echo -e "${RED}This will delete ALL data (bronze + silver + gold)${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/bronze" "Bronze layer"
            delete_dir "$DATA_DIR/silver" "Silver layer"
            delete_dir "$DATA_DIR/gold" "Gold layer"
        else
            echo "Cancelled."
        fi
        ;;
    6)
        echo ""
        echo -e "${RED}This will delete ALL weather data (all layers)${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/bronze/demand_weather" "Bronze weather data"
            delete_dir "$DATA_DIR/silver/demand_weather" "Silver weather data"
            delete_dir "$DATA_DIR/gold/demand_features" "Gold demand features"
        else
            echo "Cancelled."
        fi
        ;;
    7)
        echo ""
        echo -e "${RED}This will delete ALL EPİAŞ data (all layers)${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/bronze/epias" "Bronze EPİAŞ data"
            delete_dir "$DATA_DIR/silver/epias" "Silver EPİAŞ data"
        else
            echo "Cancelled."
        fi
        ;;
    8)
        echo ""
        echo -e "${YELLOW}Custom selection:${NC}"
        echo ""
        read -p "Delete bronze/demand_weather? (y/n): " del1
        read -p "Delete bronze/epias? (y/n): " del2
        read -p "Delete silver/demand_weather? (y/n): " del3
        read -p "Delete silver/epias? (y/n): " del4
        read -p "Delete gold/demand_features? (y/n): " del5
        echo ""
        echo -e "${RED}Confirm custom deletion?${NC}"
        read -p "Proceed? (yes/no): " confirm

        if [ "$confirm" = "yes" ]; then
            [ "$del1" = "y" ] && delete_dir "$DATA_DIR/bronze/demand_weather" "Bronze weather data"
            [ "$del2" = "y" ] && delete_dir "$DATA_DIR/bronze/epias" "Bronze EPİAŞ data"
            [ "$del3" = "y" ] && delete_dir "$DATA_DIR/silver/demand_weather" "Silver weather data"
            [ "$del4" = "y" ] && delete_dir "$DATA_DIR/silver/epias" "Silver EPİAŞ data"
            [ "$del5" = "y" ] && delete_dir "$DATA_DIR/gold/demand_features" "Gold demand features"
        else
            echo "Cancelled."
        fi
        ;;
    9)
        echo "Cancelled."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Cleanup complete!${NC}"
echo -e "${GREEN}============================================${NC}"
