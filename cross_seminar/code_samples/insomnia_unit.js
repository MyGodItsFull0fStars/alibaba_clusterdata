const response = await insomnia.send();
const body = JSON.parse(response.data);
const item = body[0];

expect(body).to.be.an('array');
expect(item).to.be.an('object');
expect(item).to.have.property('id');