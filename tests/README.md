# CLI Output Tests

Tests for validating that `phonopy-irreps` CLI output remains consistent with reference outputs.

## Running Tests

```bash
# Run all tests
pytest tests/test_cli.py -v

# Run specific test
pytest tests/test_cli.py::test_phonopy_irreps_output -v
```

## Updating References

When you intentionally change the output format (e.g., adding new columns, changing labels), update the reference files:

```bash
# Regenerate all reference outputs
pytest tests/test_cli.py --generate-references -v
```

This will:
1. Run `phonopy-irreps` on each example YAML file
2. Save output to `tests/reference/<example_name>.txt`
3. Skip the comparison test

## Reference Files

- `tests/reference/2_basic_phonopy/BaTiO3_phonopy_params.txt` - BaTiO3 reference output
- `tests/reference/3_advanced_phonopy/TmFeO3_phonopy_params.txt` - TmFeO3 reference output

## CI/CD Integration

Add to your CI workflow:

```yaml
- name: Run CLI tests
  run: pytest tests/test_cli.py -v
```

## Test Failures

If tests fail, check:
1. `test_results/` directory for diff files showing differences
2. Whether the change was intentional (update references)
3. Whether a bug was introduced (fix the code)
