#!/bin/bash
################################################################################
# ForeWatt Data Cleanup Script
# Deletes data from medallion architecture layers (Bronze, Silver, Gold)
# Updated for complete data structure with calendar, macro, and influx support
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Data directory
DATA_DIR="./data"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  ForeWatt Data Cleanup Utility${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
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

echo -e "${GREEN}┌─ Bronze Layer (Raw Data):${NC}"
echo "│  1. Calendar data: $DATA_DIR/bronze/calendar/"
show_size "$DATA_DIR/bronze/calendar"
echo "│  2. EPİAŞ data: $DATA_DIR/bronze/epias/"
show_size "$DATA_DIR/bronze/epias"
echo "│  3. Weather data: $DATA_DIR/bronze/demand_weather/"
show_size "$DATA_DIR/bronze/demand_weather"
echo "│  4. Macro data: $DATA_DIR/bronze/macro/"
show_size "$DATA_DIR/bronze/macro"
echo ""

echo -e "${GREEN}├─ Silver Layer (Normalized Data):${NC}"
echo "│  5. Calendar data: $DATA_DIR/silver/calendar/"
show_size "$DATA_DIR/silver/calendar"
echo "│  6. EPİAŞ data: $DATA_DIR/silver/epias/"
show_size "$DATA_DIR/silver/epias"
echo "│  7. Weather data: $DATA_DIR/silver/demand_weather/"
show_size "$DATA_DIR/silver/demand_weather"
echo "│  8. Macro data: $DATA_DIR/silver/macro/"
show_size "$DATA_DIR/silver/macro"
echo ""

echo -e "${GREEN}├─ Gold Layer (Feature-Engineered Data):${NC}"
echo "│  9.  Calendar features: $DATA_DIR/gold/calendar_features/"
show_size "$DATA_DIR/gold/calendar_features"
echo "│  10. EPİAŞ features: $DATA_DIR/gold/epias/"
show_size "$DATA_DIR/gold/epias"
echo "│  11. Demand features: $DATA_DIR/gold/demand_features/"
show_size "$DATA_DIR/gold/demand_features"
echo "│  12. Lag features: $DATA_DIR/gold/lag_features/"
show_size "$DATA_DIR/gold/lag_features"
echo "│  13. Rolling features: $DATA_DIR/gold/rolling_features/"
show_size "$DATA_DIR/gold/rolling_features"
echo "│  14. Master dataset: $DATA_DIR/gold/master/"
show_size "$DATA_DIR/gold/master"
echo ""

echo -e "${CYAN}├─ Database (InfluxDB):${NC}"
echo "│  15. InfluxDB data: $DATA_DIR/influx/"
show_size "$DATA_DIR/influx"
echo ""

echo -e "${CYAN}└─ Other:${NC}"
echo "   16. Unused data: $DATA_DIR/unused/"
show_size "$DATA_DIR/unused"
echo ""

if [ -d "$DATA_DIR" ]; then
    echo -e "${BLUE}Total data directory size:${NC}"
    show_size "$DATA_DIR"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}What would you like to delete?${NC}"
echo ""
echo "  ${GREEN}Full Layers:${NC}"
echo "    1) Bronze layer only (all raw data)"
echo "    2) Silver layer only (all normalized data)"
echo "    3) Gold layer only (all feature-engineered data)"
echo "    4) Bronze + Silver layers"
echo "    5) Bronze + Silver + Gold (all medallion layers)"
echo ""
echo "  ${CYAN}By Data Type:${NC}"
echo "    6) Calendar data (all layers)"
echo "    7) EPİAŞ data (all layers)"
echo "    8) Weather data (all layers)"
echo "    9) Macro data (all layers)"
echo ""
echo "  ${CYAN}Special:${NC}"
echo "   10) InfluxDB database only"
echo "   11) Unused folder only"
echo "   12) InfluxDB + Unused folders"
echo "   13) Everything except InfluxDB"
echo "   14) EVERYTHING (complete wipe)"
echo ""
echo "  ${YELLOW}Advanced:${NC}"
echo "   15) Custom selection (choose specific directories)"
echo ""
echo "   0) Cancel"
echo ""
read -p "Enter your choice (0-15): " choice

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
        echo -e "${RED}This will delete ALL medallion layers (Bronze + Silver + Gold)${NC}"
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
        echo -e "${RED}This will delete ALL calendar data (all layers)${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/bronze/calendar" "Bronze calendar data"
            delete_dir "$DATA_DIR/silver/calendar" "Silver calendar data"
            delete_dir "$DATA_DIR/gold/calendar_features" "Gold calendar features"
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
            delete_dir "$DATA_DIR/gold/epias" "Gold EPİAŞ features"
        else
            echo "Cancelled."
        fi
        ;;
    8)
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
    9)
        echo ""
        echo -e "${RED}This will delete ALL macro data (all layers)${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/bronze/macro" "Bronze macro data"
            delete_dir "$DATA_DIR/silver/macro" "Silver macro data"
        else
            echo "Cancelled."
        fi
        ;;
    10)
        echo ""
        echo -e "${RED}This will delete InfluxDB database${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/influx" "InfluxDB database"
        else
            echo "Cancelled."
        fi
        ;;
    11)
        echo ""
        echo -e "${RED}This will delete unused folder${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/unused" "Unused folder"
        else
            echo "Cancelled."
        fi
        ;;
    12)
        echo ""
        echo -e "${RED}This will delete InfluxDB + Unused folders${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/influx" "InfluxDB database"
            delete_dir "$DATA_DIR/unused" "Unused folder"
        else
            echo "Cancelled."
        fi
        ;;
    13)
        echo ""
        echo -e "${RED}This will delete EVERYTHING except InfluxDB${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            delete_dir "$DATA_DIR/bronze" "Bronze layer"
            delete_dir "$DATA_DIR/silver" "Silver layer"
            delete_dir "$DATA_DIR/gold" "Gold layer"
            delete_dir "$DATA_DIR/unused" "Unused folder"
        else
            echo "Cancelled."
        fi
        ;;
    14)
        echo ""
        echo -e "${RED}╔════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  WARNING: COMPLETE DATA WIPE          ║${NC}"
        echo -e "${RED}║  This will delete EVERYTHING!         ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════╝${NC}"
        echo ""
        read -p "Type 'DELETE EVERYTHING' to confirm: " confirm
        if [ "$confirm" = "DELETE EVERYTHING" ]; then
            delete_dir "$DATA_DIR/bronze" "Bronze layer"
            delete_dir "$DATA_DIR/silver" "Silver layer"
            delete_dir "$DATA_DIR/gold" "Gold layer"
            delete_dir "$DATA_DIR/influx" "InfluxDB database"
            delete_dir "$DATA_DIR/unused" "Unused folder"
        else
            echo "Cancelled (confirmation text did not match)."
        fi
        ;;
    15)
        echo ""
        echo -e "${YELLOW}Custom selection - Choose specific directories to delete:${NC}"
        echo ""
        echo -e "${GREEN}Bronze layer:${NC}"
        read -p "  Delete bronze/calendar? (y/n): " del_b_cal
        read -p "  Delete bronze/epias? (y/n): " del_b_epi
        read -p "  Delete bronze/demand_weather? (y/n): " del_b_wea
        read -p "  Delete bronze/macro? (y/n): " del_b_mac
        echo ""
        echo -e "${GREEN}Silver layer:${NC}"
        read -p "  Delete silver/calendar? (y/n): " del_s_cal
        read -p "  Delete silver/epias? (y/n): " del_s_epi
        read -p "  Delete silver/demand_weather? (y/n): " del_s_wea
        read -p "  Delete silver/macro? (y/n): " del_s_mac
        echo ""
        echo -e "${GREEN}Gold layer:${NC}"
        read -p "  Delete gold/calendar_features? (y/n): " del_g_cal
        read -p "  Delete gold/epias? (y/n): " del_g_epi
        read -p "  Delete gold/demand_features? (y/n): " del_g_dem
        read -p "  Delete gold/lag_features? (y/n): " del_g_lag
        read -p "  Delete gold/rolling_features? (y/n): " del_g_rol
        read -p "  Delete gold/master? (y/n): " del_g_mas
        echo ""
        echo -e "${CYAN}Other:${NC}"
        read -p "  Delete influx database? (y/n): " del_influx
        read -p "  Delete unused folder? (y/n): " del_unused
        echo ""
        echo -e "${RED}Confirm custom deletion?${NC}"
        read -p "Proceed? (yes/no): " confirm

        if [ "$confirm" = "yes" ]; then
            # Bronze
            [ "$del_b_cal" = "y" ] && delete_dir "$DATA_DIR/bronze/calendar" "Bronze calendar"
            [ "$del_b_epi" = "y" ] && delete_dir "$DATA_DIR/bronze/epias" "Bronze EPİAŞ"
            [ "$del_b_wea" = "y" ] && delete_dir "$DATA_DIR/bronze/demand_weather" "Bronze weather"
            [ "$del_b_mac" = "y" ] && delete_dir "$DATA_DIR/bronze/macro" "Bronze macro"

            # Silver
            [ "$del_s_cal" = "y" ] && delete_dir "$DATA_DIR/silver/calendar" "Silver calendar"
            [ "$del_s_epi" = "y" ] && delete_dir "$DATA_DIR/silver/epias" "Silver EPİAŞ"
            [ "$del_s_wea" = "y" ] && delete_dir "$DATA_DIR/silver/demand_weather" "Silver weather"
            [ "$del_s_mac" = "y" ] && delete_dir "$DATA_DIR/silver/macro" "Silver macro"

            # Gold
            [ "$del_g_cal" = "y" ] && delete_dir "$DATA_DIR/gold/calendar_features" "Gold calendar features"
            [ "$del_g_epi" = "y" ] && delete_dir "$DATA_DIR/gold/epias" "Gold EPİAŞ features"
            [ "$del_g_dem" = "y" ] && delete_dir "$DATA_DIR/gold/demand_features" "Gold demand features"
            [ "$del_g_lag" = "y" ] && delete_dir "$DATA_DIR/gold/lag_features" "Gold lag features"
            [ "$del_g_rol" = "y" ] && delete_dir "$DATA_DIR/gold/rolling_features" "Gold rolling features"
            [ "$del_g_mas" = "y" ] && delete_dir "$DATA_DIR/gold/master" "Gold master dataset"

            # Other
            [ "$del_influx" = "y" ] && delete_dir "$DATA_DIR/influx" "InfluxDB database"
            [ "$del_unused" = "y" ] && delete_dir "$DATA_DIR/unused" "Unused folder"
        else
            echo "Cancelled."
        fi
        ;;
    0)
        echo "Cancelled."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Cleanup complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Show remaining data size if data directory still exists
if [ -d "$DATA_DIR" ]; then
    echo -e "${BLUE}Remaining data:${NC}"
    show_size "$DATA_DIR"
    echo ""
fi
