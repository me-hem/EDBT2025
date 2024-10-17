-- 1. Create Database
CREATE DATABASE "EthData";


-- 2. Create table
CREATE TABLE BlockData(number BIGINT PRIMARY KEY, baseFeePerGas NUMERIC, gasUsed BIGINT, burntFee NUMERIC);


-- 3. Create View
CREATE OR REPLACE VIEW CumulativeAvgBurntFee AS
SELECT 
    FLOOR(COUNT(*) / 32) AS epoch,
    AVG(burntFee) AS avgBurntFee
FROM BlockData;


-- 4. Create udf
CREATE OR REPLACE FUNCTION CalculateCumulativeAvgBurntFee()
RETURNS TABLE (latestEpoch NUMERIC, updatedAvgBurntFee NUMERIC) AS $$
DECLARE
    blockCount BIGINT;
BEGIN
    SELECT COUNT(*) INTO blockCount FROM BlockData;

    IF blockCount % 32 = 0 THEN
        RETURN QUERY 
        SELECT epoch::NUMERIC AS latestEpoch, avgBurntFee::NUMERIC as updatedAvgBurntFee
        FROM CumulativeAvgBurntFee;
    ELSE
        RETURN;
    END IF;
END;
$$ LANGUAGE plpgsql;
