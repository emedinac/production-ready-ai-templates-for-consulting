-- Initialize additional databases for DVC storage
-- This script runs during PostgreSQL container initialization

-- Create DVC cache database
CREATE DATABASE dvc_cache;
GRANT ALL PRIVILEGES ON DATABASE dvc_cache TO postgres;

-- Create models database
CREATE DATABASE models;
GRANT ALL PRIVILEGES ON DATABASE models TO postgres;

-- Create data database
CREATE DATABASE data;
GRANT ALL PRIVILEGES ON DATABASE data TO postgres;

-- Create user for DVC operations (optional, for security)
-- CREATE USER dvc_user WITH PASSWORD 'dvc_password';
-- GRANT CONNECT ON DATABASE dvc_cache TO dvc_user;
-- GRANT CONNECT ON DATABASE models TO dvc_user;
-- GRANT CONNECT ON DATABASE data TO dvc_user;