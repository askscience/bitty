pub fn redact_secret_args(args: &[String]) -> Vec<String> {
    let mut redacted = Vec::with_capacity(args.len());
    let mut redact_next = false;
    for arg in args {
        if redact_next {
            redacted.push("<redacted>".into());
            redact_next = false;
            continue;
        }
        if arg == "--cluster-token" || arg == "--token" {
            redacted.push(arg.clone());
            redact_next = true;
        } else if let Some((key, _)) = arg.split_once('=') {
            if key == "--cluster-token" || key == "--token" {
                redacted.push(format!("{key}=<redacted>"));
            } else {
                redacted.push(redact_iroh_token(arg));
            }
        } else {
            redacted.push(redact_iroh_token(arg));
        }
    }
    redacted
}

fn redact_iroh_token(value: &str) -> String {
    if let Some((prefix, rest)) = value.split_once("token=") {
        let suffix = rest
            .split_once('&')
            .map(|(_, suffix)| format!("&{suffix}"))
            .unwrap_or_default();
        format!("{prefix}token=<redacted>{suffix}")
    } else {
        value.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_cluster_token_flag_value() {
        let args = vec!["node".into(), "--cluster-token".into(), "secret".into()];
        assert_eq!(
            redact_secret_args(&args),
            vec!["node", "--cluster-token", "<redacted>"]
        );
    }

    #[test]
    fn redacts_iroh_token_query_parameter() {
        let args = vec!["iroh://abc?token=secret&relay=https://relay/".into()];
        assert_eq!(
            redact_secret_args(&args),
            vec!["iroh://abc?token=<redacted>&relay=https://relay/"]
        );
    }
}
