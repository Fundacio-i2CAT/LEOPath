#!/bin/bash

# LEO Routing Simulation Docker Launcher
# This script provides an easy way to run the LEO routing simulation using Docker Compose

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] -c [CONFIG_FILE]"
    echo ""
    echo "Commands:"
    echo "  run           Run the LEO routing simulation with specified config"
    echo "  visualise     Generate visualization and start HTTP server"
    echo "  clean         Clean all generated files and tear down containers"
    echo ""
    echo "Options:"
    echo "  -c, --config  Config file path (required for run/visualise commands)"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 run -c src/config/ether.yaml"
    echo "  $0 visualise -c src/config/ether_simple.yaml"
    echo "  $0 clean"
}

# Function to check if Docker and Docker Compose are installed
check_dependencies() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! command -v docker compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
}

# Function to create output directories
create_output_dirs() {
    print_status "Creating output directories..."
    mkdir -p output/{logs,tles,simulation,visualizations}
    print_success "Output directories created"
}

# Function to validate config file
validate_config() {
    local config_file="$1"
    
    if [[ -z "$config_file" ]]; then
        print_error "Config file is required. Use -c option."
        show_usage
        exit 1
    fi
    
    if [[ ! -f "$config_file" ]]; then
        print_error "Config file not found: $config_file"
        exit 1
    fi
    
    print_status "Using config file: $config_file"
}

# Function to run simulation
run_simulation() {
    local config_file="$1"
    
    validate_config "$config_file"
    create_output_dirs
    
    print_status "Starting LEO routing simulation..."
    print_status "Config: $config_file"
    
    # Run the simulation container with the config file
    docker compose run --rm leo-routing-simu "$config_file"
    
    print_success "Simulation completed successfully"
}

# Function to run visualization
run_visualization() {
    local config_file="$1"
    
    validate_config "$config_file"
    create_output_dirs
    
    print_status "Generating visualization..."
    print_status "Config: $config_file"
    
    # Generate visualization
    docker compose run --rm --entrypoint "python -m src.satellite_visualisation.cesium_builder.main" leo-routing-viz "$config_file"
    
    print_success "Visualization generated successfully"
    
    # Start the HTTP server
    print_status "Starting HTTP server for visualization..."
    docker compose up -d viz-server
    
    print_success "HTTP server started"
    print_status "Visualization available at: http://localhost:8080"
    print_status "Press Ctrl+C to stop the server, or run 'docker compose down' to clean up"
}

# Function to clean all generated files and containers
clean_all() {
    print_warning "This will delete ALL generated files and tear down Docker containers:"
    echo "  - All files in ./logs/"
    echo "  - All files in ./output/"
    echo "  - All files in ./src/satellite_visualisation/visualisation_output/"
    echo "  - All files in ./src/satellite_visualisation/cesium_builder/visualisation_output/"
    echo "  - All simulation log files (*.log)"
    echo "  - All Docker containers and networks"
    echo ""
    
    # Prompt for confirmation
    read -p "Are you sure you want to proceed? This action cannot be undone! (y/N): " -n 1 -r
    echo  # Move to a new line
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Clean operation cancelled"
        return 0
    fi
    
    print_status "Starting cleanup process..."
    
    # Tear down Docker containers
    print_status "Stopping and removing Docker containers..."
    docker compose down --volumes --remove-orphans 2>/dev/null || true
    
    # Remove Docker images (optional - commented out to preserve build cache)
    # print_status "Removing Docker images..."
    # docker compose down --rmi all 2>/dev/null || true
    
    # Clean generated files
    print_status "Removing generated files..."
    
    # Remove logs directory contents
    if [[ -d "./logs" ]]; then
        rm -rf ./logs/*
        print_status "Cleaned ./logs/"
    fi
    
    # Remove output directory contents
    if [[ -d "./output" ]]; then
        rm -rf ./output/*
        print_status "Cleaned ./output/"
    fi
    
    # Remove visualization output
    if [[ -d "./src/satellite_visualisation/visualisation_output" ]]; then
        rm -rf ./src/satellite_visualisation/visualisation_output/*
        print_status "Cleaned ./src/satellite_visualisation/visualisation_output/"
    fi
    
    # Remove cesium builder visualization output
    if [[ -d "./src/satellite_visualisation/cesium_builder/visualisation_output" ]]; then
        rm -rf ./src/satellite_visualisation/cesium_builder/visualisation_output/*
        print_status "Cleaned ./src/satellite_visualisation/cesium_builder/visualisation_output/"
    fi
    
    # Remove simulation log files
    rm -f ./simulation_*.log 2>/dev/null || true
    print_status "Removed simulation log files"
    
    # Remove coverage files
    rm -f ./cov.xml 2>/dev/null || true
    rm -rf ./.coverage* 2>/dev/null || true
    print_status "Removed coverage files"
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    print_status "Removed Python cache files"
    
    print_success "Cleanup completed successfully!"
    print_status "All generated files have been removed and containers have been torn down"
}

# Main script logic
main() {
    # Check dependencies first
    check_dependencies
    
    # Handle empty arguments
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 0
    fi
    
    # Parse command and options
    local command=""
    local config_file=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            run|visualise|clean)
                command="$1"
                shift
                ;;
            -c|--config)
                config_file="$2"
                shift 2
                ;;
            -h|--help|help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo ""
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Execute command
    case "$command" in
        run)
            run_simulation "$config_file"
            ;;
        visualise)
            run_visualization "$config_file"
            ;;
        clean)
            clean_all
            ;;
        "")
            print_error "No command specified"
            echo ""
            show_usage
            exit 1
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run the main function with all arguments
main "$@"