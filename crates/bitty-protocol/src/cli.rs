use std::fmt::Display;
use std::str::FromStr;

pub fn required_next(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {name}"))
}

pub fn parse_next<T>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T, String>
where
    T: FromStr,
    T::Err: Display,
{
    let value = required_next(args, name)?;
    value
        .parse()
        .map_err(|err| format!("invalid value for {name}: {err}"))
}

pub fn required_next_or_exit(args: &mut impl Iterator<Item = String>, name: &str) -> String {
    required_next(args, name).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(2);
    })
}

pub fn parse_next_or_exit<T>(args: &mut impl Iterator<Item = String>, name: &str) -> T
where
    T: FromStr,
    T::Err: Display,
{
    parse_next(args, name).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(2);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_next_reports_missing_values() {
        let mut args = Vec::<String>::new().into_iter();
        assert_eq!(
            required_next(&mut args, "--flag"),
            Err("missing value for --flag".into())
        );
    }

    #[test]
    fn parse_next_reports_invalid_values() {
        let mut args = vec!["abc".to_string()].into_iter();
        assert_eq!(
            parse_next::<u32>(&mut args, "--count"),
            Err("invalid value for --count: invalid digit found in string".into())
        );
    }
}
