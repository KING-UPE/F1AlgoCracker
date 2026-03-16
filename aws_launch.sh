#!/bin/bash
# ============================================================
#  AWS CLI Launch Commands — F1 Constants Cracker
#  Run these from your LOCAL machine (not EC2)
#
#  Prerequisites:
#    brew install awscli  (macOS)   OR   pip install awscli
#    aws configure        (set your access key + region)
# ============================================================

# ── 1. CONFIGURATION — edit these ────────────────────────────
REGION="us-east-1"          # Cheapest region for compute
KEY_NAME="your-key-pair"    # Your EC2 key pair name (for SSH)
YOUR_IP=$(curl -s https://checkip.amazonaws.com)/32   # Your IP for SSH access

# ── 2. FIND LATEST AMAZON LINUX 2023 AMI ─────────────────────
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=al2023-ami-*-x86_64" \
        "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)

echo "Using AMI: $AMI_ID"

# ── 3. CREATE SECURITY GROUP ─────────────────────────────────
SG_ID=$(aws ec2 create-security-group \
    --region "$REGION" \
    --group-name "f1-cracker-sg" \
    --description "F1 Constants Cracker" \
    --query "GroupId" --output text)

# Allow SSH from your IP only
aws ec2 authorize-security-group-ingress \
    --region "$REGION" \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr "$YOUR_IP"

echo "Security group: $SG_ID"

# ── 4. ENCODE USER DATA ───────────────────────────────────────
# Edit REPO_URL in aws_userdata.sh first!
USER_DATA=$(base64 < aws_userdata.sh)

# ── 5. OPTION A: SPOT INSTANCE (cheapest, ~$0.60-1.20/hr) ────
# c6i.32xlarge: 128 vCPUs, 256 GB RAM
# Best for: differential_evolution with workers=-1
echo "Requesting spot instance..."

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "c6i.32xlarge" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
    --user-data "$USER_DATA" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=f1-cracker}]' \
    --query "Instances[0].InstanceId" \
    --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Waiting for instance to start..."

aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)

echo ""
echo "============================================"
echo "  Instance running at: $PUBLIC_IP"
echo "============================================"
echo ""
echo "Monitor progress:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP} 'tail -f ~/crack.log'"
echo ""
echo "Copy constants when done:"
echo "  scp -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}:~/box-box-box/solution/constants.json solution/"
echo ""
echo "Terminate instance when done (IMPORTANT — stops billing):"
echo "  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"

# ── 5B. OPTION B: ON-DEMAND (guaranteed, no interruption risk) ─
# Uncomment to use on-demand instead:
#
# INSTANCE_ID=$(aws ec2 run-instances \
#     --region "$REGION" \
#     --image-id "$AMI_ID" \
#     --instance-type "c6i.32xlarge" \   # $5.44/hr on-demand
#     --key-name "$KEY_NAME" \
#     --security-group-ids "$SG_ID" \
#     --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":50}}]' \
#     --user-data "$USER_DATA" \
#     --query "Instances[0].InstanceId" --output text)

# ── INSTANCE TYPE COMPARISON ──────────────────────────────────
# ┌─────────────────┬────────┬──────────┬──────────┬──────────┐
# │ Instance        │  vCPUs │   RAM    │ Spot/hr  │ OD/hr    │
# ├─────────────────┼────────┼──────────┼──────────┼──────────┤
# │ c6i.8xlarge     │    32  │  64 GB   │ ~$0.20   │ $1.36    │
# │ c6i.16xlarge    │    64  │ 128 GB   │ ~$0.40   │ $2.72    │
# │ c6i.32xlarge ✓  │   128  │ 256 GB   │ ~$0.80   │ $5.44    │
# │ c7i.48xlarge    │   192  │ 384 GB   │ ~$1.20   │ $8.50    │
# │ hpc6id.32xlarge │   128  │ 512 GB   │  N/A     │ $5.67    │
# └─────────────────┴────────┴──────────┴──────────┴──────────┘
#
# RECOMMENDATION: c6i.32xlarge spot
#   - 128 vCPUs → differential_evolution uses all cores
#   - ~$0.80/hr spot → 10 min runtime = ~$0.13 total cost
#   - Interruption risk is low for short jobs

# ── SPOT PRICE CHECK ─────────────────────────────────────────
echo ""
echo "Current spot prices (c6i.32xlarge across AZs):"
aws ec2 describe-spot-price-history \
    --region "$REGION" \
    --instance-types "c6i.32xlarge" \
    --product-descriptions "Linux/UNIX" \
    --max-items 5 \
    --query "SpotPriceHistory[*].[AvailabilityZone,SpotPrice]" \
    --output table
