def _get_keycloak_open_id() -> KeycloakOpenID:
    """
    Creates a KeyCloak ID instance that is required to authenticate 
    our own service to other services and also to authenticate 
    other services.

    Returns:
        KeycloakOpenID: Instance of a KeyCloak ID
    """
    return KeycloakOpenID(
        server_url='https://datacloud-auth.euprojects.net/auth/',
        client_id=__get_client_id(),
        realm_name='user-authentication',
        client_secret_key=__get_secret_key()
    )