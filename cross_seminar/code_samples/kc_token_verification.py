def verify_keycloak_token(keycloak_token: dict) -> bool:
    """Verify a given KeyCloak token

    Args:
        keycloak_token (dict): a KeyCloak token to be verified

    Returns:
        bool:  returns True if the token could be verified by KeyCloak, else returns False
    """
    if keycloak_token is None or len(keycloak_token) == 0:
        return False
    if 'access_token' not in keycloak_token:
        return False
    try:
        keycloak_response = __keycloak_open_id.userinfo(
            keycloak_token['access_token'])

        if 'email_verified' not in keycloak_response or 'preferred_username' not in keycloak_response:
            return False

        return True

    except KeycloakAuthenticationError as err:
        # Token could not be verified
        print('Invalid Access Token', err)
        return False