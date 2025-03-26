{ ... }:

{
  overlays = [(
    finalPkgs: prevPkgs: {
      # NOP
    }
  )];

  config = {
    allowUnfree = true;
    glibc.withLdFallbackPatch = true;
  };
}
