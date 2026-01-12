-- Create Keyspace
CREATE KEYSPACE IF NOT EXISTS sentinel_vision 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE sentinel_vision;

-- Table for Raw Stream Metadata
CREATE TABLE IF NOT EXISTS stream_metadata (
    stream_id UUID PRIMARY KEY,
    source_url text,
    resolution text,
    fps int,
    start_time timestamp,
    status text
);

-- Table for High-Frequency Inference Results
-- Optimized for time-series retrieval
CREATE TABLE IF NOT EXISTS inference_results (
    stream_id UUID,
    timestamp timestamp,
    object_class text,
    confidence float,
    bounding_box frozen<list<float>>,
    metadata map<text, text>,
    PRIMARY KEY (stream_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);
