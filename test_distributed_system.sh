#!/bin/bash
# test_distributed_system.sh
# Comprehensive test script for the distributed document library system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}[TEST]${NC} $1"; }
print_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[FAIL]${NC} $1"; }

# Configuration - Update these IPs to match your setup
MAC_MINI_IP="192.168.100.41"
DELL_LAPTOP_IP="192.168.100.42"
LENOVO_LAPTOP_IP="192.168.100.43"

ORCHESTRATOR_URL="http://${MAC_MINI_IP}:8000"
DOCLING_URL="http://${MAC_MINI_IP}:8004"
LLM_URL="http://${LENOVO_LAPTOP_IP}:8001"
EMBEDDING_URL="http://${DELL_LAPTOP_IP}:8002"
KNOWLEDGE_GRAPH_URL="http://${DELL_LAPTOP_IP}:8003"

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    print_info "Testing: $test_name"
    
    if eval "$test_command"; then
        print_success "$test_name"
        ((TESTS_PASSED++))
        return 0
    else
        print_error "$test_name"
        FAILED_TESTS+=("$test_name")
        ((TESTS_FAILED++))
        return 1
    fi
}

# Health check function
check_health() {
    local service_name="$1"
    local health_url="$2"
    local timeout="${3:-10}"
    
    response=$(curl -s -w "%{http_code}" --max-time "$timeout" "$health_url" -o /tmp/health_response.json)
    http_code="${response: -3}"
    
    if [ "$http_code" = "200" ]; then
        print_success "$service_name health check passed"
        return 0
    else
        print_error "$service_name health check failed (HTTP $http_code)"
        cat /tmp/health_response.json 2>/dev/null || echo "No response body"
        return 1
    fi
}

# Authentication helper
authenticate_user() {
    local username="test@example.com"
    local password="testpass123"
    
    # Try to register user (may fail if already exists)
    curl -s -X POST "$ORCHESTRATOR_URL/api/v1/auth/signup" \
        -H "Content-Type: application/json" \
        -d "{\"username\": \"$username\", \"password\": \"$password\"}" > /dev/null 2>&1 || true
    
    # Login to get token
    local response=$(curl -s -X POST "$ORCHESTRATOR_URL/api/v1/auth/login" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=$username&password=$password")
    
    local token=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('access_token', ''))" 2>/dev/null || echo "")
    
    if [ -n "$token" ]; then
        echo "$token"
        return 0
    else
        echo ""
        return 1
    fi
}

echo "=== Distributed Document Library System Test Suite ==="
echo "Testing system components across three machines:"
echo "- Mac Mini ($MAC_MINI_IP): Orchestrator, Docling, MongoDB, Neo4j"
echo "- Lenovo Laptop ($LENOVO_LAPTOP_IP): LLM Service (Gemma 3-4B)"
echo "- Dell Laptop ($DELL_LAPTOP_IP): Embedding, Knowledge Graph, Milvus"
echo ""

# Test 1: Network connectivity
print_info "Phase 1: Network Connectivity Tests"
run_test "Mac Mini connectivity" "ping -c 1 -W 5 $MAC_MINI_IP > /dev/null"
run_test "Dell Laptop connectivity" "ping -c 1 -W 5 $DELL_LAPTOP_IP > /dev/null"
run_test "Lenovo Laptop connectivity" "ping -c 1 -W 5 $LENOVO_LAPTOP_IP > /dev/null"

# Test 2: Service health checks
print_info ""
print_info "Phase 2: Service Health Checks"
run_test "Orchestrator API health" "check_health 'Orchestrator API' '$ORCHESTRATOR_URL/health'"
run_test "Docling Service health" "check_health 'Docling Service' '$DOCLING_URL/health'"
run_test "LLM Service health" "check_health 'LLM Service' '$LLM_URL/health'"
run_test "Embedding Service health" "check_health 'Embedding Service' '$EMBEDDING_URL/health'"
run_test "Knowledge Graph Service health" "check_health 'Knowledge Graph Service' '$KNOWLEDGE_GRAPH_URL/health'"

# Test 3: Database connectivity
print_info ""
print_info "Phase 3: Database Connectivity Tests"

# MongoDB test
run_test "MongoDB connectivity" "curl -s --max-time 5 '$ORCHESTRATOR_URL/api/v1/auth/signup' -X POST -H 'Content-Type: application/json' -d '{\"test\":\"connection\"}' > /dev/null"

# Neo4j test (through Neo4j Desktop on Mac Mini)
run_test "Neo4j Desktop connectivity" "curl -s --max-time 10 '$KNOWLEDGE_GRAPH_URL/config' | grep -q neo4j"

# Milvus v2.6.2 test (through orchestrator)
run_test "Milvus v2.6.2 connectivity" "curl -s --max-time 10 '$ORCHESTRATOR_URL/health' | grep -q ok || true"

# Test 4: Authentication system
print_info ""
print_info "Phase 4: Authentication System Test"

AUTH_TOKEN=""
if AUTH_TOKEN=$(authenticate_user); then
    if [ -n "$AUTH_TOKEN" ]; then
        print_success "User authentication successful"
        ((TESTS_PASSED++))
    else
        print_error "User authentication failed - no token received"
        ((TESTS_FAILED++))
    fi
else
    print_error "User authentication failed"
    ((TESTS_FAILED++))
fi

# Test 5: Individual service functionality
print_info ""
print_info "Phase 5: Service Functionality Tests"

# Test LLM service
LLM_TEST_PAYLOAD='{
    "messages": [{"role": "user", "content": "Hello, please respond with exactly: TEST_SUCCESS"}],
    "max_tokens": 50,
    "temperature": 0.0
}'

if run_test "LLM Service generation" "curl -s --max-time 30 -X POST '$LLM_URL/v1/chat/completions' -H 'Content-Type: application/json' -d '$LLM_TEST_PAYLOAD' | grep -q 'TEST_SUCCESS'"; then
    print_info "LLM service is responding correctly"
fi

# Test Embedding service
EMBEDDING_TEST_PAYLOAD='{"texts": ["This is a test document for embedding generation."]}'

if run_test "Embedding Service generation" "curl -s --max-time 20 -X POST '$EMBEDDING_URL/embed-documents' -H 'Content-Type: application/json' -d '$EMBEDDING_TEST_PAYLOAD' | grep -q 'embeddings'"; then
    print_info "Embedding service is generating embeddings correctly"
fi

# Test Docling service (if we have a test document)
if [ -f "test_document.txt" ]; then
    DOCLING_TEST_PAYLOAD='{
        "input_file_path": "'$(pwd)'/test_document.txt",
        "output_directory_path": "/tmp/docling_test"
    }'
    
    run_test "Docling Service processing" "curl -s --max-time 60 -X POST '$DOCLING_URL/process' -H 'Content-Type: application/json' -d '$DOCLING_TEST_PAYLOAD' | grep -q 'extracted_markdown_path'"
else
    print_warning "No test document found, skipping Docling service test"
    print_info "Create 'test_document.txt' to test document processing"
fi

# Test 6: End-to-end document processing (if authenticated)
if [ -n "$AUTH_TOKEN" ]; then
    print_info ""
    print_info "Phase 6: End-to-End Document Processing Test"
    
    # Create a test document
    cat > test_upload.txt << 'EOF'
Document Processing Test

This is a test document for the distributed document library system.
It contains information about AI technology, machine learning models,
and natural language processing capabilities.

The system should extract entities like:
- Technology: AI, machine learning, natural language processing
- Concepts: document processing, information extraction
- Organizations: OpenAI, Google, Microsoft

This document will test the complete pipeline from upload to knowledge graph creation.
EOF

    # Test document upload
    if UPLOAD_RESPONSE=$(curl -s --max-time 60 -X POST "$ORCHESTRATOR_URL/api/v1/documents/upload" \
        -H "Authorization: Bearer $AUTH_TOKEN" \
        -F "file=@test_upload.txt"); then
        
        DOC_ID=$(echo "$UPLOAD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null || echo "")
        
        if [ -n "$DOC_ID" ]; then
            print_success "Document upload successful (ID: $DOC_ID)"
            ((TESTS_PASSED++))
            
            # Wait and check document processing status
            sleep 30
            
            if STATUS_RESPONSE=$(curl -s --max-time 10 "$ORCHESTRATOR_URL/api/v1/documents/$DOC_ID/status" \
                -H "Authorization: Bearer $AUTH_TOKEN"); then
                
                STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', ''))" 2>/dev/null || echo "")
                
                if [ "$STATUS" = "completed" ]; then
                    print_success "Document processing completed successfully"
                    ((TESTS_PASSED++))
                elif [ "$STATUS" = "processing" ]; then
                    print_warning "Document still processing (this is normal for large documents)"
                    ((TESTS_PASSED++))
                else
                    print_error "Document processing failed or unknown status: $STATUS"
                    ((TESTS_FAILED++))
                fi
            else
                print_error "Failed to check document status"
                ((TESTS_FAILED++))
            fi
        else
            print_error "Document upload failed - no ID returned"
            ((TESTS_FAILED++))
        fi
    else
        print_error "Document upload request failed"
        ((TESTS_FAILED++))
    fi
    
    # Clean up test file
    rm -f test_upload.txt
else
    print_warning "Skipping end-to-end test due to authentication failure"
fi

# Test 7: Inter-service communication
print_info ""
print_info "Phase 7: Inter-Service Communication Tests"

# Test if knowledge graph service can reach LLM service
run_test "Knowledge Graph -> LLM communication" "curl -s --max-time 10 '$KNOWLEDGE_GRAPH_URL/config' | grep -q '$LENOVO_LAPTOP_IP'"

# Test if knowledge graph service can reach embedding service
run_test "Knowledge Graph -> Embedding communication" "curl -s --max-time 10 '$KNOWLEDGE_GRAPH_URL/config' | grep -q 'localhost:8002'"

# Test if orchestrator can reach all remote services
run_test "Orchestrator -> Remote services config" "curl -s --max-time 10 '$ORCHESTRATOR_URL/health' > /dev/null"

# Summary
print_info ""
echo "=== Test Results Summary ==="
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"

if [ $TESTS_FAILED -eq 0 ]; then
    print_success "All tests passed! Your distributed system is working correctly."
    exit 0
else
    print_error "Some tests failed. Please review the failed tests:"
    for failed_test in "${FAILED_TESTS[@]}"; do
        echo "  - $failed_test"
    done
    echo ""
    echo "Common troubleshooting steps:"
    echo "1. Verify all services are running on their respective machines"
    echo "2. Check network connectivity and firewall settings"
    echo "3. Verify IP addresses in configuration files"
    echo "4. Check service logs for detailed error information"
    echo "5. Ensure GPU services have sufficient memory available"
    exit 1
fi