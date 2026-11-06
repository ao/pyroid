#!/usr/bin/env bash
# Mac-side control of the pyroid EC2 run (bench/ec2/README.md). bash 3.2 compatible (macOS /bin/bash); needs aws and jq.
#
#   ec2.sh preflight                    read-only checks: identity, SG has no ingress, VPC, subnets, AMIs, profile, bucket
#   ec2.sh stage                        bundle `next` (+ feat/ec2bench for the tooling overlay) and upload it, the scripts
#                                       (committed versions) and /tmp/pyroid-ec2bench/pins/ to the bucket
#   ec2.sh launch x86|arm               run-instances (c7i.4xlarge | c8g.4xlarge) with every required guard; logs the row
#   ec2.sh wait-ssm ID                  until SSM reports the instance Online
#   ec2.sh bootstrap ID                 send-command: fetch scripts, bootstrap.sh (foreground, executionTimeout 3600 s)
#   ec2.sh start ID [VAR=value ...]     send-command: run.sh as the systemd unit pyroid-run (detached; resumable)
#   ec2.sh exec ID 'command'            send-command for an ad-hoc look (prints its output)
#   ec2.sh invocation CMD ID            status and output of a send-command
#   ec2.sh progress [MACHINE...]        status.json and the run.log tail from the bucket
#   ec2.sh watch [SECONDS]              poll progress every SECONDS [300] until every machine is done/failed
#   ec2.sh collect MACHINE              download + unpack results to /tmp/pyroid-ec2bench/collect/MACHINE
#   ec2.sh terminate ID                 terminate-instances, only for IDs this script launched (registry + tags)
#   ec2.sh verify                       describe-instances for every launched ID
#   ec2.sh s3-clean                     delete everything under s3://<bucket>/ec2bench/ (our objects only)
#
# Hard rules (the brief): profile primary1, region us-west-2 on every call, not overridable; at most 2 instances; only
# c7i.4xlarge/c8g.4xlarge; SG sg-<redacted> (no ingress) only; never touch anything not launched here.
set -euo pipefail

AWS_ARGS=(--profile primary1 --region us-west-2)
BUCKET=${PYROID_BENCH_BUCKET:?set PYROID_BENCH_BUCKET to an S3 bucket you own}
S3=s3://$BUCKET/ec2bench
SG=sg-<redacted>
VPC=vpc-<redacted>
INSTANCE_PROFILE=pyroid-bench-profile
LEDGER=/Users/devel/src/ao/pyroid/.orchestration/aws-resources.md
LOCAL=/tmp/pyroid-ec2bench
REGISTRY=$LOCAL/instances.tsv
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
mkdir -p "$LOCAL"

aws_() { aws "${AWS_ARGS[@]}" "$@"; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
die() { echo "ec2.sh: $*" >&2; exit 1; }
itype() { case $1 in x86) echo c7i.4xlarge ;; arm) echo c8g.4xlarge ;; *) die "arch must be x86 or arm" ;; esac; }
amiarch() { case $1 in x86) echo x86_64 ;; arm) echo arm64 ;; esac; }
machine_of() { case $1 in x86) echo aws-c7i.4xlarge ;; arm) echo aws-c8g.4xlarge ;; *) echo "$1" ;; esac; }

registered() { [[ -f $REGISTRY ]] && cut -f1 "$REGISTRY" | grep -qx "$1"; }
check_ours() {  # ID: launched by this script, and tagged as such
    [[ $1 =~ ^i-[0-9a-f]{8,17}$ ]] || die "not an instance id: $1"
    registered "$1" || die "$1 is not in $REGISTRY (only instances launched by ec2.sh)"
    local tags
    tags=$(aws_ ec2 describe-instances --instance-ids "$1" --query 'Reservations[0].Instances[0].Tags' --output json)
    [[ $(jq -r '.[] | select(.Key=="project") | .Value' <<<"$tags") == pyroid-bench ]] || die "$1 lacks tag project=pyroid-bench"
    [[ $(jq -r '.[] | select(.Key=="Name") | .Value' <<<"$tags") == pyroid-bench-* ]] || die "$1 lacks Name=pyroid-bench-*"
}

cmd_preflight() {
    aws_ sts get-caller-identity --query '[Account,Arn]' --output text
    local ingress
    ingress=$(aws_ ec2 describe-security-groups --group-ids "$SG" --query 'length(SecurityGroups[0].IpPermissions)' --output text)
    [[ $ingress == 0 ]] || die "$SG has $ingress ingress rules; it must have none"
    echo "SG $SG: no ingress rules"
    aws_ ec2 describe-vpcs --vpc-ids "$VPC" --query 'Vpcs[0].[VpcId,IsDefault]' --output text
    aws_ ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC" Name=default-for-az,Values=true \
        --query 'Subnets[].[SubnetId,AvailabilityZone,MapPublicIpOnLaunch]' --output text
    local a ami
    for a in x86 arm; do
        ami=$(aws_ ssm get-parameter --name "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-$(amiarch $a)" \
            --query Parameter.Value --output text)
        echo "$a $(itype $a): $ami $(aws_ ec2 describe-images --image-ids "$ami" --query 'Images[0].[Name,RootDeviceName]' --output text)"
    done
    aws_ iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" --query 'InstanceProfile.Roles[].RoleName' --output text
    echo "bucket:"; aws_ s3 ls "$S3/" --recursive | tail -n 20 || true
    echo "instances tagged project=pyroid-bench:"
    aws_ ec2 describe-instances --filters Name=tag:project,Values=pyroid-bench \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime]' --output text
}

cmd_stage() {
    local st=$LOCAL/stage
    rm -rf "$st"
    mkdir -p "$st"
    git -C "$REPO" rev-parse --verify -q next >/dev/null || die "no branch next"
    echo "next @ $(git -C "$REPO" rev-parse --short next); feat/ec2bench @ $(git -C "$REPO" rev-parse --short feat/ec2bench)"
    git -C "$REPO" bundle create "$LOCAL/pyroid-next.bundle" next feat/ec2bench
    git -C "$REPO" bundle verify "$LOCAL/pyroid-next.bundle"
    # The committed scripts, not the working tree: what runs on the instance is what the branch says.
    git -C "$REPO" archive feat/ec2bench bench/ec2 | tar -x -C "$st"
    aws_ s3 cp --only-show-errors "$LOCAL/pyroid-next.bundle" "$S3/code/pyroid-next.bundle"
    aws_ s3 cp --only-show-errors --recursive "$st/bench/ec2/" "$S3/code/ec2/"
    if [[ -d $LOCAL/pins ]]; then aws_ s3 cp --only-show-errors --recursive "$LOCAL/pins/" "$S3/pins/"; fi
    aws_ s3 ls "$S3/" --recursive
}

ledger_add() {  # id type name
    printf '| ec2 instance %s (%s, ec2bench) | %s | %s | |\n' "$2" "$3" "$1" "$(now)" >>"$LEDGER"
}
ledger_removed() {  # id
    local t; t=$(now)
    if grep -q "^| ec2 instance .*| $1 | [^|]* | |$" "$LEDGER"; then
        sed -i '' "s#^\(| ec2 instance .*| $1 | [^|]* |\) |\$#\1 $t |#" "$LEDGER"
    else
        printf '| ec2 instance (ec2bench) | %s | | %s |\n' "$1" "$t" >>"$LEDGER"
    fi
}

cmd_launch() {
    local arch=${1:?x86|arm} t name ami subnet n id ud
    t=$(itype "$arch")
    name=pyroid-bench-$(amiarch "$arch")
    n=$(aws_ ec2 describe-instances --filters Name=tag:project,Values=pyroid-bench \
        Name=instance-state-name,Values=pending,running,stopping,stopped --query 'length(Reservations[].Instances[])' --output text)
    ((n < 2)) || die "$n pyroid-bench instances exist; the limit is 2 at a time"
    [[ -z $(aws_ ec2 describe-instances --filters "Name=tag:Name,Values=$name" Name=instance-state-name,Values=pending,running \
        --query 'Reservations[].Instances[].InstanceId' --output text) ]] || die "a $name instance is already running"
    [[ $(aws_ ec2 describe-security-groups --group-ids "$SG" --query 'length(SecurityGroups[0].IpPermissions)' --output text) == 0 ]] ||
        die "$SG has ingress rules"
    ami=$(aws_ ssm get-parameter --name "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-$(amiarch "$arch")" \
        --query Parameter.Value --output text)
    subnet=${SUBNET:-$(aws_ ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC" Name=default-for-az,Values=true \
        "Name=availability-zone,Values=$(aws_ ec2 describe-instance-type-offerings --location-type availability-zone \
            --filters "Name=instance-type,Values=$t" --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' ',')" \
        --query 'sort_by(Subnets,&AvailabilityZone)[0].SubnetId' --output text)}
    [[ $subnet == subnet-* ]] || die "no default subnet found"
    ud=${USER_DATA:-$LOCAL/stage/bench/ec2/user-data.sh}
    [[ -f $ud ]] || die "run ec2.sh stage first ($ud missing)"
    [[ $(sed -n 2p "$ud") == "shutdown -h +420"* ]] || die "user data must start with shutdown -h +420"
    local dry=() out note=""
    if [[ ${DRY_RUN:-0} == 1 ]]; then dry=(--dry-run); note=" (dry run: nothing is created)"; fi
    echo "launching $t ($name) from $ami in $subnet$note"
    out=$(aws_ ec2 run-instances ${dry[@]+"${dry[@]}"} \
        --client-token "pyroid-ec2bench-$arch-$(date -u +%Y%m%d)-${ATTEMPT:-1}" \
        --image-id "$ami" --instance-type "$t" --count 1 \
        --subnet-id "$subnet" --security-group-ids "$SG" --associate-public-ip-address \
        --iam-instance-profile "Name=$INSTANCE_PROFILE" \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=1 \
        --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=project,Value=pyroid-bench},{Key=Name,Value=$name}]" \
        "ResourceType=volume,Tags=[{Key=project,Value=pyroid-bench},{Key=Name,Value=$name}]" \
        --user-data "file://$ud" \
        --query 'Instances[0].InstanceId' --output text 2>&1) || true
    if ((${#dry[@]})); then
        [[ $out == *DryRunOperation* ]] || die "dry run failed: $out"
        echo "dry run ok: $out"
        return 0
    fi
    id=$out
    [[ $id == i-* ]] || die "run-instances returned '$id'"
    registered "$id" || { printf '%s\t%s\t%s\t%s\n' "$id" "$t" "$arch" "$(now)" >>"$REGISTRY"; ledger_add "$id" "$t" "$name"; }
    echo "$id"
}

cmd_wait_ssm() {
    local id=${1:?ID} s n=0
    while ((n++ < 60)); do
        s=$(aws_ ssm describe-instance-information --filters "Key=InstanceIds,Values=$id" \
            --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
        echo "$(date -u +%T) $id SSM: $s"
        [[ $s == Online ]] && return 0
        sleep 20
    done
    die "$id not Online after 20 min"
}

send() {  # id comment execution-timeout-s script -> command id
    local params
    params=$(jq -n --arg s "$4" --arg t "$3" '{commands: ($s | split("\n")), executionTimeout: [$t]}')
    aws_ ssm send-command --instance-ids "$1" --document-name AWS-RunShellScript --comment "pyroid-ec2bench $2" \
        --timeout-seconds 600 --parameters "$params" \
        --output-s3-bucket-name "$BUCKET" --output-s3-key-prefix ec2bench/ssm \
        --query Command.CommandId --output text
}

cmd_invocation() {  # cmd id: waits until the command finishes
    local c=${1:?CMD} id=${2:?ID} s
    while :; do
        s=$(aws_ ssm get-command-invocation --command-id "$c" --instance-id "$id" --query Status --output text 2>/dev/null || echo Pending)
        case $s in Pending | InProgress | Delayed) sleep "${POLL:-30}" ;; *) break ;; esac
    done
    echo "status: $s"
    aws_ ssm get-command-invocation --command-id "$c" --instance-id "$id" \
        --query '[StandardOutputContent,StandardErrorContent]' --output text | tail -n "${TAIL:-60}"
    [[ $s == Success ]]
}

cmd_bootstrap() {
    local id=${1:?ID}
    check_ours "$id"
    local c
    c=$(send "$id" bootstrap 3600 "set -eu
mkdir -p /opt/pyroid/ec2
aws s3 cp --only-show-errors --recursive $S3/code/ec2/ /opt/pyroid/ec2/
bash /opt/pyroid/ec2/bootstrap.sh >/opt/pyroid/bootstrap.out 2>&1 || { tail -n 80 /opt/pyroid/bootstrap.out; exit 1; }
tail -n 40 /opt/pyroid/bootstrap.out")
    echo "command $c"
    POLL=30 cmd_invocation "$c" "$id"
}

cmd_start() {
    local id=${1:?ID}
    shift
    check_ours "$id"
    local envs="--setenv=PYROID_S3=$S3 --setenv=HOME=/root" kv c
    for kv in "$@"; do
        case $kv in
            [A-Z_]*=*) ;;
            *) die "expected VAR=value: $kv" ;;
        esac
        case $kv in *[!A-Za-z0-9_=.,:/+\ -]*) die "VAR=value may only use [A-Za-z0-9_=.,:/+ -]: $kv" ;; esac
        envs="$envs '--setenv=$kv'"
    done
    c=$(send "$id" start 600 "set -eu
mkdir -p /opt/pyroid/ec2
aws s3 cp --only-show-errors --recursive $S3/code/ec2/ /opt/pyroid/ec2/
if systemctl is-active --quiet pyroid-run; then echo 'pyroid-run is already active'; systemctl status pyroid-run --no-pager | head -n 6; exit 0; fi
systemctl reset-failed pyroid-run 2>/dev/null || true
systemd-run --unit=pyroid-run --description='pyroid ec2bench run' $envs /bin/bash /opt/pyroid/ec2/run.sh
sleep 10
systemctl status pyroid-run --no-pager | head -n 12")
    echo "command $c"
    POLL=10 cmd_invocation "$c" "$id"
}

cmd_exec() {
    local id=${1:?ID} c
    check_ours "$id"
    c=$(send "$id" exec 600 "${2:?command}")
    POLL=5 cmd_invocation "$c" "$id"
}

cmd_progress() {
    local m
    [[ $# -gt 0 ]] || set -- aws-c7i.4xlarge aws-c8g.4xlarge
    for m in "$@"; do
        m=$(machine_of "$m")
        echo "== $m $(aws_ s3 cp "$S3/results/$m/status.json" - 2>/dev/null || echo '(no status yet)')"
        aws_ s3 cp "$S3/results/$m/run.log" - 2>/dev/null | tail -n "${TAIL:-8}" || true
    done
}

cmd_watch() {
    local every=${1:-300} m s all
    while :; do
        all=1
        for m in aws-c7i.4xlarge aws-c8g.4xlarge; do
            s=$(aws_ s3 cp "$S3/results/$m/status.json" - 2>/dev/null || echo '{}')
            echo "$(date -u +%T) $m $(jq -c '{state, step, done, failed, skipped, loadavg}' <<<"$s" 2>/dev/null || echo "$s")"
            [[ $(jq -r '.state // ""' <<<"$s" 2>/dev/null) == "done" || $(jq -r '.state // ""' <<<"$s" 2>/dev/null) == "failed" ]] || all=0
        done
        ((all)) && { echo "every machine finished"; return 0; }
        sleep "$every"
    done
}

cmd_collect() {
    local m; m=$(machine_of "${1:?MACHINE}")
    local d=$LOCAL/collect/$m
    rm -rf "$d"
    mkdir -p "$d"
    aws_ s3 cp --only-show-errors --recursive "$S3/results/$m/" "$d/"
    (cd "$d" && zstd -dc results.tar.zst | tar -x)
    [[ -f $d/refs-new.tar.zst ]] && (cd "$d" && zstd -dc refs-new.tar.zst | tar -x)
    find "$d" -maxdepth 3 | head -n 40
}

cmd_terminate() {
    local id=${1:?ID}
    check_ours "$id"
    aws_ ec2 terminate-instances --instance-ids "$id" --query 'TerminatingInstances[0].[InstanceId,PreviousState.Name,CurrentState.Name]' --output text
    ledger_removed "$id"
}

cmd_verify() {
    [[ -f $REGISTRY ]] || die "no instances launched"
    local id rest
    while IFS=$'\t' read -r id rest; do
        echo "$id $(aws_ ec2 describe-instances --instance-ids "$id" \
            --query 'Reservations[0].Instances[0].[InstanceType,State.Name,LaunchTime,StateTransitionReason]' --output text </dev/null)"
    done <"$REGISTRY"
}

cmd_s3_clean() {
    aws_ s3 rm --only-show-errors --recursive "$S3/"
    echo "left under $S3/: $(aws_ s3 ls "$S3/" --recursive | wc -l | tr -d ' ') objects"
}

sub=${1:-}
[[ -n $sub ]] || { sed -n '2,19p' "$0"; exit 2; }
shift
case $sub in
    preflight | stage | launch | bootstrap | start | exec | invocation | progress | watch | collect | terminate | verify) "cmd_$sub" "$@" ;;
    wait-ssm) cmd_wait_ssm "$@" ;;
    s3-clean) cmd_s3_clean "$@" ;;
    *) die "unknown subcommand $sub" ;;
esac
