#!/bin/bash

# LEO Routing Simulator Docker Helper Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CONFIG_FILE="src/config/ether_simple.yaml"
ACTION="run"

# Help function
show_help() {
    echo "LEO Routing Simulator Docker Helper"
    echo ""
    echo "Usage: $0 [OPTIONS] [ACTION]"
    echo ""
    echo "ACTIONS:"
    echo "  run         Run simulation (default)"
    echo "  build       Build Docker image"
    echo "  visualize   Generate visualization"
    echo "  serve       Start web server for visualizations"
    echo "  clean       Clean up containers and volumes"
    echo ""
    echo "OPTIONS:"
    echo "  -c, --config FILE    Configuration file (default: $CONFIG_FILE)"
    echo "  -h, --help          Show this help"
    echo ""
    echo "OUTPUT ORGANIZATION:"
    echo "  All outputs are organized in the output/ directory:"
    echo "    output/logs/           - Simulation log files"
    echo "    output/simulation/     - Simulation results"  
    echo "    output/tles/          - Generated TLE files"
    echo "    output/visualizations/ - HTML visualizations"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 build                                    # Build Docker image"
    echo "  $0 run                                      # Run with default config"
    echo "  $0 -c src/config/ether.yaml run           # Run with full constellation"
    echo "  $0 visualize                               # Generate visualization"
    echo "  $0 serve                                   # Start web server on port 8080"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        build|run|visualize|serve|clean)
            ACTION="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Create output directories
mkdir -p logs tle_output visualisation_output

case $ACTION in
    build)
        echo -e "${GREEN}Building Docker image...${NC}"
        docker build -t leo-routing-simu .
        ;;
    
    run)
        echo -e "${GREEN}Running simulation with config: $CONFIG_FILE${NC}"
        docker run --rm \
            -v "$(pwd)/src/config:/app/src/config:ro" \
            -v "$(pwd)/logs:/app/logs" \
            -v "$(pwd)/tle_output:/app/tle_output" \
            -v "$(pwd):/app/output" \
            -w /app/output \
            leo-routing-simu "$CONFIG_FILE"
        echo -e "${GREEN}Simulation complete! Check current directory and logs/ for results.${NC}"
        ;;
    
    visualize)
        echo -e "${GREEN}Generating visualization with config: $CONFIG_FILE${NC}"
        docker run --rm \
            -v "$(pwd)/src/config:/app/src/config:ro" \
            -v "$(pwd)/output/tles:/app/tle_output" \
            -v "$(pwd)/output/visualizations:/app/visualisation_output" \
            -v "$(pwd):/app/output" \
            -w /app/output \
            --entrypoint python \
            leo-routing-simu -m src.satellite_visualisation.cesium_builder.main "$CONFIG_FILE"
        echo -e "${GREEN}Visualization generated! Check output/visualizations/ directory.${NC}"
        ;;
    
    serve)
        echo -e "${GREEN}Starting web server on http://localhost:8080${NC}"
        echo -e "${YELLOW}Visualizations available at: http://localhost:8080/output/visualizations/${NC}"
        docker run --rm \
            -p 8080:80 \
            -v "$(pwd):/usr/share/nginx/html:ro" \
            nginx:alpine
        ;;
    
    clean)
        echo -e "${YELLOW}Cleaning up Docker containers and volumes...${NC}"
        docker compose down -v 2>/dev/null || true
        docker container prune -f
        echo -e "${GREEN}Cleanup complete!${NC}"
        ;;
    
    *)
        echo -e "${RED}Unknown action: $ACTION${NC}"
        show_help
        exit 1
        ;;
esac
