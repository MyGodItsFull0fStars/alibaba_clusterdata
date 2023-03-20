def _get_keycloak_token(username: str, password: str) -> dict:
    """Get the assigned KeyCloak token for the ADA-PIPE backend to be able to send requests to other DataCloud services.

    Returns:
        dict: the keycloak token for ADA-PIPE
    """
    if username is None or len(username) == 0:
        raise KeycloakAuthenticationError('The provided username is either None or empty')
    if password is None or len(password) == 0:
        raise KeycloakAuthenticationError('The provided password is either None or empty')
    
    token = __keycloak_open_id.token(
        username,
        password)
    return token